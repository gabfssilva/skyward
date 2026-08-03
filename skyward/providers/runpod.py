import asyncio
import re
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Self

import httpx
import msgspec

from skyward.shared.accelerators import CATALOG, resolve
from skyward.shared.errors import CapabilityMismatchError
from skyward.shared.provider import Binding, Machine, Mount
from skyward.shared.providers import RunPod
from skyward.shared.schemas import ComputeSpec, Market, Offer, Volume
from skyward.worker import bootstrap

API_URL = "https://api.runpod.io/v2"
CLOUDS: tuple[str, ...] = ("SECURE", "COMMUNITY")

DEFAULT_IMAGE = "nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04"
DEFAULT_PORTS: tuple[str, ...] = ("22/tcp",)

BASE_IMAGE_FALLBACKS: dict[str, str] = {
    "nvidia": "nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04",
    "runpod-base": "runpod/base:1.0.3-cuda1290-ubuntu2204",
    "runpod-pytorch": "runpod/pytorch:2.8.0-py3.13-cuda12.8.1-devel-ubuntu24.04",
}

BASE_IMAGE_REPOS: dict[str, str] = {
    "nvidia": "nvidia/cuda",
    "runpod-base": "runpod/base",
    "runpod-pytorch": "runpod/pytorch",
}

DOCKER_HUB_URL = "https://hub.docker.com"
CUDA_OPEN_UPPER: tuple[int, int] = (99, 9)

_CUDA_DOTTED = re.compile(r"cuda(\d+)\.(\d+)")
_CUDA_COMPACT = re.compile(r"cu(?:da)?(\d{2})(\d)\d")
_TAG_VERSION = re.compile(r"^(\d+)\.(\d+)\.(\d+)")
_UBUNTU = re.compile(r"ubuntu(\d{2})\.?(\d{2})")
_NVIDIA_VARIANT = re.compile(r"cudnn\d*-runtime")

KNOWN_COUNTRIES: tuple[str, ...] = (
    "US", "CA", "DE", "FR", "NL", "SE", "CZ", "RO", "IS", "NO", "DK", "GB", "JP", "IN", "SG", "AU",
)

DEADSWITCH = (
    'if [ "${INSTANCE_TIMEOUT:-0}" -gt 0 ] 2>/dev/null; then '
    "( idle=0; while :; do "
    'if grep -qsE ":0016[[:space:]]+[0-9A-Fa-f]+:[0-9A-Fa-f]+[[:space:]]+01[[:space:]]" '
    "/proc/net/tcp /proc/net/tcp6; then idle=0; "
    'else idle=$((idle+15)); [ "$idle" -ge "$INSTANCE_TIMEOUT" ] && '
    '{ runpodctl remove pod "$RUNPOD_POD_ID" 2>/dev/null || kill 1; }; fi; '
    "sleep 15; done ) & fi; "
)
"""A self-terminating watchdog, armed only when ``INSTANCE_TIMEOUT`` is set above zero.

RunPod bills a pod for as long as it exists, and a pod whose deploy response was lost —
created, but never written down — is one nothing in the control plane will ever
terminate. This is the net under that: the daemon holds an SSH connection to the pod
for the pool's whole life, so an established connection on port 22 (``:0016``, state
``01`` in ``/proc/net/tcp``) is the liveness signal. Once it has been gone for
``INSTANCE_TIMEOUT`` seconds the pod removes itself with ``runpodctl`` — deleting the
pod outright so not even its disk keeps billing — and falls back to killing PID 1 where
``runpodctl`` is absent. It never fires while the pool is attached, so a run may take as
long as it takes."""

ENTRYPOINT = (
    DEADSWITCH
    + "([ -x /usr/sbin/sshd ] || ("
    "apt-get -o DPkg::Lock::Timeout=-1 update && "
    "DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=-1 "
    "install -y --no-install-recommends openssh-server"
    ")) && "
    "mkdir -p /run/sshd ~/.ssh && "
    'echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys && '
    "chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys && "
    "ssh-keygen -A && "
    'sed -i "s/#PermitRootLogin.*/PermitRootLogin yes/" /etc/ssh/sshd_config && '
    "/usr/sbin/sshd && sleep infinity"
)


class RunPodProvider:
    """RunPod — a two-tier GPU cloud: SECURE (datacenter fleet) and COMMUNITY (peer hosts).

    Its catalog is a small fixed list of GPU types, but ``lowestPrice`` tracks live
    host supply and interruptible bids, so the TTL is minutes rather than hours.
    """

    kind: ClassVar[str] = "runpod"
    credential_fields: ClassVar[tuple[str, ...]] = ("api_key",)
    offers_ttl: ClassVar[timedelta] = timedelta(minutes=10)

    def allows_cluster_formation(self, spec: ComputeSpec, offer: Offer) -> bool:
        return offer.specific.get("cloud_type") != "COMMUNITY" and self._config.global_networking is True

    def __init__(self, provider_id: str, name: str, api_key: str, config: RunPod) -> None:
        self._id = provider_id
        self._name = name
        self._api_key = api_key
        self._config = config

    @classmethod
    def create(cls, provider_id: str, name: str, credentials: Mapping[str, str], config: Mapping[str, Any]) -> Self:
        settings = msgspec.convert({**credentials, **config}, RunPod)
        if not settings.api_key:
            raise CapabilityMismatchError("runpod requires an api_key credential", provider=name)
        return cls(provider_id, name, settings.api_key, settings)

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    @property
    def _timeout(self) -> int:
        return self._config.request_timeout

    def _image(self, spec: ComputeSpec) -> str:
        """The image to run, in precedence order.

        A ``sky.Image(base=...)`` at the pool is the user's most explicit word and
        wins; then the provider's own ``container_image`` override; then the family
        picked by ``base_image``; then the built-in default.
        """
        return spec.image.base or self._config.container_image or BASE_IMAGE_FALLBACKS.get(self._config.base_image, DEFAULT_IMAGE)

    async def _image_candidates(self, spec: ComputeSpec, offer: Offer) -> tuple[str, ...]:
        """The images to try, newest supported CUDA first.

        An explicit choice — a ``sky.Image(base=...)`` or the provider's
        ``container_image`` — is one image and the deploy tries only it. Left to the
        ``base_image`` family, the newest Docker Hub tag is picked for each CUDA the
        accelerator supports, so a host that refuses the top one is retried against the
        next rather than failing the launch. The static fallback answers when Docker Hub
        is unreachable or the accelerator has no known CUDA range.
        """
        if spec.image.base or self._config.container_image:
            return (self._image(spec),)

        base_image = self._config.base_image
        repo = BASE_IMAGE_REPOS.get(base_image)
        cuda_min, cuda_max = _cuda_range(offer.accelerator)
        if repo is None or cuda_min is None:
            return (self._image(spec),)

        tags = await _fetch_docker_tags(repo, self._timeout)
        variant = _NVIDIA_VARIANT if base_image == "nvidia" else re.compile(r".")
        candidates = _select_image_candidates(
            tags,
            _cuda_pair(cuda_min),
            _cuda_pair(cuda_max) if cuda_max else CUDA_OPEN_UPPER,
            self._config.ubuntu,
            repo,
            variant,
        )
        return candidates or (self._image(spec),)

    def _countries(self) -> tuple[str, ...]:
        """Allowed country codes after exclusions; empty means no constraint."""
        excluded = self._config.excluded
        match self._config.countries:
            case () if not excluded:
                return ()
            case ():
                return tuple(c for c in KNOWN_COUNTRIES if c not in excluded)
            case allowed:
                return tuple(c for c in allowed if c not in excluded)

    def _data_center(self) -> str | None:
        centers = self._config.centers
        return centers[0] if centers else None

    async def _data_centers(self, client: httpx.AsyncClient) -> tuple[str, ...]:
        countries = self._countries()
        if not countries:
            return ()
        response = await client.get(f"{API_URL}/catalog/datacenters", headers=self._headers)
        response.raise_for_status()
        centers = (response.json() or {}).get("dataCenters") or []
        allowed = frozenset(countries)
        return tuple(
            str(center["id"])
            for center in centers
            if allowed.intersection(str(center["id"]).split("-"))
        )

    async def _registry_auth_id(self, client: httpx.AsyncClient) -> str | None:
        """Resolve the named registry credential to its id, ``None`` if unset or absent."""
        name = self._config.registry_auth
        if not name:
            return None
        response = await client.get(f"{API_URL}/registries", headers=self._headers)
        response.raise_for_status()
        creds = (response.json() or {}).get("registries") or []
        return next((c["id"] for c in creds if str(c.get("name", "")).lower() == str(name).lower()), None)

    async def _gpu_types(self, client: httpx.AsyncClient, cloud: str) -> list[dict[str, Any]]:
        response = await client.get(
            f"{API_URL}/catalog/gpus",
            params={"include": "AVAILABILITY", "product": "POD", "cloud": cloud},
            headers=self._headers,
        )
        response.raise_for_status()
        return (response.json() or {}).get("gpus") or []

    async def offers(self) -> AsyncIterator[Offer]:
        wanted = self._config.cloud_type.upper()
        clouds = tuple(cloud for cloud in CLOUDS if cloud == wanted) or CLOUDS

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            catalogs = await asyncio.gather(*(self._gpu_types(client, cloud) for cloud in clouds))

        now = datetime.now(UTC)
        expires_at = now + self.offers_ttl

        for cloud, gpu_types in zip(clouds, catalogs, strict=True):
            tier = cloud.lower()
            for gpu in gpu_types:
                if not gpu.get(tier):
                    continue

                price = gpu.get("price") or {}
                on_demand = price.get(tier)
                if on_demand is None:
                    continue

                gpu_id = str(gpu["id"])
                display = str(gpu.get("name") or gpu_id)
                vram_gb = int(gpu.get("memory") or 0)
                max_count = int((gpu.get("maxCount") or {}).get(tier) or 1)
                available = 0 if gpu.get("availability") == "NONE" else None

                for count in range(1, max_count + 1):
                    yield Offer(
                        id=f"{gpu_id}:{cloud}:{count}x",
                        provider_id=self._id,
                        provider_name=self._name,
                        kind=self.kind,
                        billing_unit="hour",
                        instance_type=f"{gpu_id}:{cloud}",
                        accelerator=display,
                        accelerator_count=count,
                        vram=float(vram_gb) or None,
                        cpus=0,
                        memory_gb=0,
                        spot_price=None,
                        on_demand_price=round(on_demand * count, 4) if on_demand is not None else None,
                        available=available,
                        fetched_at=now,
                        expires_at=expires_at,
                        specific={
                            "gpu_type_id": gpu_id,
                            "cloud_type": cloud,
                            "gpu_display_name": display,
                            "gpu_memory_gb": vram_gb,
                            "stock_status": gpu.get("availability"),
                        },
                    )

    async def initialize(self, compute_id: str, spec: ComputeSpec, offer: Offer, market: Market, public_key: str) -> Binding:
        """Resolve the pod recipe. RunPod has nothing to create up front.

        There is no keypair resource, no network and no security group: the public
        key travels as a ``PUBLIC_KEY`` env var and the entrypoint appends it to
        ``authorized_keys`` on first boot. Everything here is therefore a decision
        rather than an allocation, which is what makes it trivially idempotent —
        and what makes :meth:`release` a no-op.
        """
        gpu_count = offer.accelerator_count
        if market == "spot" and not offer.spot_price:
            raise CapabilityMismatchError(
                f"{offer.id} was picked on the spot market and carries no bid price", provider=self._name,
            )
        if self._config.cluster_mode != "individual":
            raise CapabilityMismatchError(
                "RunPod REST v2 beta does not expose Instant Cluster creation",
                provider=self._name,
            )
        if self._config.cpu_clock != "3c":
            raise CapabilityMismatchError(
                "RunPod REST v2 beta does not expose GPU host CPU-clock filtering",
                provider=self._name,
            )
        if self._config.min_inet_down is not None or self._config.min_inet_up is not None:
            raise CapabilityMismatchError(
                "RunPod REST v2 beta does not expose pod network-speed filtering",
                provider=self._name,
            )

        data_center_id = self._data_center() or next(
            (s.region for s in spec.specs if s.provider.kind == self.kind and s.region), None,
        )
        countries = self._countries()
        multiplier = self._config.bid_multiplier

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            registry_auth_id = await self._registry_auth_id(client)
            country_centers = await self._data_centers(client)
        candidates = await self._image_candidates(spec, offer)
        data_center_ids = (str(data_center_id),) if data_center_id else country_centers

        return {
            "compute_id": compute_id,
            "prefix": f"skyward-{compute_id}-",
            "image": candidates[0],
            "image_candidates": list(candidates),
            "public_key": public_key,
            "ttl": spec.ttl,
            "gpu_type_id": offer.specific["gpu_type_id"],
            "gpu_count": gpu_count,
            "cloud_type": offer.specific["cloud_type"],
            "data_center_id": data_center_ids[0] if len(data_center_ids) == 1 else None,
            "data_center_ids": list(data_center_ids),
            "countries": list(countries),
            "country_code": countries[0] if countries else None,
            "container_disk_gb": self._config.container_disk_gb,
            "volume_gb": self._config.volume_gb,
            "volume_mount_path": self._config.volume_mount_path,
            "ports": ",".join(self._config.ports),
            "registry_auth_id": registry_auth_id,
            "global_networking": (
                self._config.global_networking
                if self._config.global_networking is not None
                else bool(spec.options.cluster)
            ),
            "bid_per_gpu": round(offer.spot_price * multiplier / gpu_count, 4) if offer.spot_price else None,
        }

    async def launch(self, binding: Binding, market: Market, count: int, min_count: int) -> tuple[Binding, Sequence[Machine]]:
        async with httpx.AsyncClient(timeout=self._timeout) as client, asyncio.TaskGroup() as group:
            attempts = [group.create_task(self._deploy(client, binding, market)) for _ in range(count)]

        results = [task.result() for task in attempts]
        machines = tuple(result for result in results if isinstance(result, Machine))
        if len(machines) < min_count:
            raise ExceptionGroup(
                f"runpod deployed {len(machines)} pods, {min_count} were required",
                [result for result in results if isinstance(result, Exception)],
            )
        return binding, machines

    async def machines(self, binding: Binding) -> Mapping[str, Machine]:
        """Find the compute's pods by name.

        RunPod has no tags or labels on a pod: the name is the only field an
        adapter controls and can filter an account-wide listing on, so the
        compute id is carried in it.
        """
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(f"{API_URL}/pods", headers=self._headers)
            response.raise_for_status()
            pods = (response.json() or {}).get("pods") or []

        found = (
            _machine(pod)
            for pod in pods
            if str(pod.get("name") or "").startswith(binding["prefix"])
        )
        return {machine.id: machine for machine in found if machine is not None}

    async def terminate(self, binding: Binding, machine_ids: tuple[str, ...]) -> None:
        if not machine_ids:
            return

        async with httpx.AsyncClient(timeout=self._timeout) as client, asyncio.TaskGroup() as group:
            for machine_id in machine_ids:
                group.create_task(self._destroy(client, machine_id))

    async def mount(self, binding: Binding, volumes: tuple[Volume, ...]) -> Mount:
        """Attach one network volume, because a pod cannot FUSE-mount anything.

        A RunPod pod runs without ``CAP_SYS_ADMIN``, so geesefs has no mount syscall
        available to it and no S3 bucket can be projected into the filesystem. What
        RunPod has instead is a network volume the host attaches before the container
        starts — so ``Volume.bucket`` is read here as the id or the name of one, and
        the phases are symlinks into where the host already put it.

        A pod takes exactly one, which is why several buckets are refused rather than
        silently reduced to the first. Several *volumes* are fine: they become
        different prefixes of the one attachment.
        """
        buckets = {volume.bucket for volume in volumes}
        if len(buckets) > 1:
            raise CapabilityMismatchError(
                f"a runpod pod attaches one network volume, not {len(buckets)}: {', '.join(sorted(buckets))}. "
                "Name one volume as bucket= and separate the datasets with prefix=",
                provider=self._name,
            )

        path = str(binding.get("volume_mount_path") or "/workspace")
        return Mount(
            binding_patch={"network_volume_id": await self._network_volume(buckets.pop())},
            phases=(bootstrap.symlinks(volumes, path),),
        )

    async def _network_volume(self, ref: str) -> str:
        """The volume's id, given either its id or the name a person gave it.

        The data centre is checked here and not left to the launch: RunPod refuses to
        attach a volume from another one with ``network volume not found``, which
        reads as a volume that does not exist rather than as one that is in the wrong
        place.
        """
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(f"{API_URL}/network-volumes", headers=self._headers)
            response.raise_for_status()
            volumes = (response.json() or {}).get("networkVolumes") or []

        matched = next((volume for volume in volumes if ref in (volume.get("id"), volume.get("name"))), None)
        if matched is None:
            available = ", ".join(str(volume.get("name") or volume.get("id")) for volume in volumes) or "none"
            raise CapabilityMismatchError(
                f"runpod has no network volume {ref!r}. Available: {available}", provider=self._name,
            )

        centre = matched.get("dataCenter")
        pinned = self._data_center()
        if centre and pinned and centre != pinned:
            raise CapabilityMismatchError(
                f"runpod network volume {ref!r} lives in {centre} but this provider is pinned to {pinned}",
                provider=self._name,
            )
        return str(matched["id"])

    async def release(self, binding: Binding) -> None:
        """Terminate every pod still carrying the compute's name prefix.

        A compute's own nodes are terminated as they are dropped; this is the sweep
        under that. A pod created in a launch whose response was lost has a name but no
        row, so nothing else will ever find it — and RunPod bills a pod for as long as
        it exists. The name is the compute id, so the pod is found here and stopped when
        the compute is released. Idempotent: a second release finds nothing left.
        """
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(f"{API_URL}/pods", headers=self._headers)
            response.raise_for_status()
            orphans = tuple(
                str(pod["id"])
                for pod in ((response.json() or {}).get("pods") or [])
                if str(pod.get("name") or "").startswith(binding["prefix"])
            )
            if orphans:
                async with asyncio.TaskGroup() as group:
                    for pod_id in orphans:
                        group.create_task(self._destroy(client, pod_id))

    async def _deploy(self, client: httpx.AsyncClient, binding: Binding, market: Market) -> Machine | Exception:
        """Deploy one pod, trying each image candidate, and hand a failure back rather than raise it.

        A launch of ``count`` pods is allowed to come back with fewer, so one host
        refusing the bid must not cancel its siblings — which is what raising inside the
        task group would do. The candidates are ordered newest-CUDA first; a host that
        has no driver for the top image is retried against the next before the pod is
        counted as lost.
        """
        candidates = binding.get("image_candidates") or [binding["image"]]
        error: Exception = RuntimeError("runpod produced no image candidate to deploy")

        for image in candidates:
            deploy = _deploy_input(binding, market, image)
            try:
                response = await client.post(f"{API_URL}/pods", json=deploy, headers=self._headers)
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPError as exc:
                error = exc
                continue

            match data:
                case {"id": str(pod_id)}:
                    return Machine(id=pod_id, state="pending")
                case _:
                    error = RuntimeError(f"runpod REST create returned no pod: {data}")

        return error

    async def _destroy(self, client: httpx.AsyncClient, machine_id: str) -> None:
        response = await client.delete(f"{API_URL}/pods/{machine_id}", headers=self._headers)
        if response.status_code != 404:
            response.raise_for_status()


def _deploy_input(binding: Binding, market: Market, image: str | None = None) -> dict[str, Any]:
    """Build a PodCreateInput from RunPod's official REST OpenAPI.

    Reads the knob keys with defaults rather than by subscript: a binding is
    persisted and a compute outlives the code that bound it, so one written before
    a knob existed must still launch under the version that added it.

    A binding from before country lists existed falls back to its lone
    ``country_code``.
    """
    if market == "spot":
        raise CapabilityMismatchError(
            "RunPod REST v2 beta does not expose spot pod creation",
            provider="runpod",
        )
    ports = binding.get("ports", ",".join(DEFAULT_PORTS))
    normalized_ports = ports.split(",") if isinstance(ports, str) else list(ports)

    deploy: dict[str, Any] = {
        "name": f"{binding['prefix']}{uuid.uuid4().hex[:8]}",
        "image": image or binding["image"],
        "args": f"bash -c '{ENTRYPOINT}'",
        "gpu": {"id": binding["gpu_type_id"], "count": binding["gpu_count"]},
        "cloud": binding["cloud_type"],
        "disk": binding["container_disk_gb"],
        "ports": normalized_ports,
        "env": {
            "PUBLIC_KEY": binding["public_key"],
            "INSTANCE_TIMEOUT": str(binding.get("ttl", 0)),
        },
    }
    path = str(binding.get("volume_mount_path", "/workspace"))
    if network_volume_id := binding.get("network_volume_id"):
        deploy["mounts"] = {"network": [{"volumeId": network_volume_id, "path": path}]}
    elif volume_gb := binding.get("volume_gb", 0):
        deploy["mounts"] = {"persistent": {"size": volume_gb, "path": path}}

    for field, value in (
        (
            "dataCenterIds",
            binding.get("data_center_ids")
            or ([center] if (center := binding.get("data_center_id")) else None),
        ),
        ("registry", binding.get("registry_auth_id")),
        ("globalNetworking", binding.get("global_networking")),
    ):
        if value is not None:
            deploy[field] = value

    return deploy


def _cuda_pair(version: str) -> tuple[int, int]:
    major, minor, *_ = version.split(".")
    return int(major), int(minor)


def _cuda_range(accelerator: str | None) -> tuple[str | None, str | None]:
    """The CUDA the accelerator supports, from the catalog, ``(None, None)`` if unknown."""
    name, _ = resolve(accelerator)
    entry = CATALOG.get(name) if name else None
    if entry and entry.cuda_min:
        return entry.cuda_min, entry.cuda_max or None
    return None, None


async def _fetch_docker_tags(repo: str, timeout: int) -> list[str]:
    """Every tag on a Docker Hub repository, newest first, empty if it cannot be read."""
    namespace, name = repo.split("/")
    tags: list[str] = []
    path: str | None = f"/v2/repositories/{namespace}/{name}/tags/"
    params: dict[str, str] | None = {"page_size": "100", "ordering": "-last_updated"}

    try:
        async with httpx.AsyncClient(base_url=DOCKER_HUB_URL, timeout=timeout) as client:
            for _ in range(50):
                response = await client.get(path or "", params=params)
                if response.status_code >= 400:
                    break
                payload = response.json()
                tags.extend(tag["name"] for tag in payload.get("results", []))
                if not (nxt := payload.get("next")):
                    break
                path, params = str(nxt).removeprefix(DOCKER_HUB_URL), None
    except httpx.HTTPError:
        return tags

    return tags


def _select_image_candidates(
    tags: list[str],
    cuda_min: tuple[int, int],
    cuda_max: tuple[int, int],
    ubuntu: str,
    repo: str,
    variant: re.Pattern[str],
) -> tuple[str, ...]:
    """The best tag for each CUDA minor within range, newest CUDA first.

    For each distinct CUDA ``major.minor`` in range, the highest patch (and newest
    Ubuntu) wins; the results are ordered highest-CUDA first so a deploy walks them
    top-down until a host accepts one.
    """
    best: dict[tuple[int, int], tuple[tuple[int, int, int], tuple[int, int], str]] = {}
    for tag in tags:
        if "ubuntu" not in tag or tag.endswith("-test") or "-dev-" in tag:
            continue
        if not _ubuntu_matches(tag, ubuntu) or not variant.search(tag):
            continue
        matched = (
            (_TAG_VERSION.match(tag) if repo == "nvidia/cuda" else None)
            or _CUDA_DOTTED.search(tag)
            or _CUDA_COMPACT.search(tag)
        )
        if not matched:
            continue
        cuda = (int(matched.group(1)), int(matched.group(2)))
        if not cuda_min <= cuda <= cuda_max:
            continue
        rank = (_extract_tag_version(tag), _extract_ubuntu(tag))
        if (prev := best.get(cuda)) is None or rank > (prev[0], prev[1]):
            best[cuda] = (*rank, tag)

    return tuple(f"{repo}:{entry[2]}" for _, entry in sorted(best.items(), reverse=True))


def _ubuntu_matches(tag: str, ubuntu: str) -> bool:
    return ubuntu == "newest" or f"ubuntu{ubuntu.replace('.', '')}" in tag or f"ubuntu{ubuntu}" in tag


def _extract_ubuntu(tag: str) -> tuple[int, int]:
    return (int(m.group(1)), int(m.group(2))) if (m := _UBUNTU.search(tag)) else (0, 0)


def _extract_tag_version(tag: str) -> tuple[int, int, int]:
    m = _TAG_VERSION.match(tag)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else (0, 0, 0)


def _machine(pod: Mapping[str, Any]) -> Machine | None:
    """A pod that has stopped is reported as gone.

    A stopped pod cannot be made to run again by anything the control plane does,
    and it keeps billing its disk for as long as it exists. Reporting it absent is
    what turns it into a lost node, and a lost node is terminated.
    """
    match pod.get("status"):
        case "EXITED" | "ERROR" | "TERMINATED":
            return None
        case _:
            runtime = pod.get("runtime") or {}
            ports = runtime.get("ports") or []
            ssh = next((port for port in ports if port.get("private") == 22), {})
            host = ssh.get("ip") or None
            networking = pod.get("globalNetworking") or {}
            return Machine(
                id=pod["id"],
                state="running" if host else "pending",
                host=host,
                port=int(ssh.get("public") or 22),
                private_host=networking.get("internalDns") or networking.get("ip") or None,
            )
