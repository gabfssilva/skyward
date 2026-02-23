from __future__ import annotations

import asyncio
import re
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

from skyward.accelerators import Accelerator
from skyward.accelerators.catalog import SPECS
from skyward.api import PoolSpec
from skyward.api.model import Cluster, Instance, InstanceStatus, InstanceType, Offer
from skyward.infra.http import HttpClient
from skyward.observability.logger import logger
from skyward.providers.provider import MountEndpoint, Provider
from skyward.providers.ssh_keys import get_local_ssh_key, get_ssh_key_path

from .client import RunPodClient, RunPodError, get_api_key
from .config import CloudType, RunPod
from .types import (
    ClusterCreateParams,
    CpuPodCreateParams,
    PodResponse,
    get_ssh_port,
)

log = logger.bind(provider="runpod")

_CUDA_DOTTED_RE = re.compile(r"cuda(\d+)\.(\d+)")
_CUDA_COMPACT_RE = re.compile(r"cu(?:da)?(\d{2})(\d)\d")
_UBUNTU_RE = re.compile(r"ubuntu(\d{2})\.?(\d{2})")
_TAG_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)")
_FALLBACK_IMAGE = "runpod/base:1.0.3-cuda1290-ubuntu2204"
_RUNPOD_S3_DATACENTERS = frozenset({
    "EUR-IS-1", "EUR-NO-1", "EU-RO-1", "EU-CZ-1",
    "US-CA-2", "US-GA-2", "US-KS-2", "US-MD-1",
    "US-MO-2", "US-NC-1", "US-NC-2",
})
_TOKEN_SPLIT = re.compile(r"[\s\-/]+")
_MIN_GPU_MATCH = 0.5


def _gpu_match_score(requested: str, display: str, gpu_id: str) -> float:
    """Score how well a GPU matches the requested name (0.0–1.0).

    Tokenizes both the query and each GPU name, then for every query token
    finds the best-matching GPU token via SequenceMatcher. The final score
    is the *minimum* best-token-match, so every part of the query must
    match well.

    "A40"  vs "NVIDIA A40"       → SM("A40","A40")=1.0   → score 1.0
    "A40"  vs "NVIDIA RTX A4000" → SM("A40","A4000")=0.75 → score 0.75
    """
    req_tokens = [t for t in _TOKEN_SPLIT.split(requested.upper()) if t]
    if not req_tokens:
        return 0.0

    best = 0.0
    for name in (display, gpu_id):
        if not name:
            continue
        name_tokens = [t for t in _TOKEN_SPLIT.split(name.upper()) if t]
        if not name_tokens:
            continue
        token_scores = [
            max(SequenceMatcher(None, rt, nt).ratio() for nt in name_tokens)
            for rt in req_tokens
        ]
        best = max(best, min(token_scores))

    return best
_DOCKER_HUB_URL = "https://hub.docker.com"
_DOCKER_HUB_REPO = "runpod/base"


def _parse_cuda_version(version: str) -> tuple[int, int]:
    major, minor, *_ = version.split(".")
    return (int(major), int(minor))


def _extract_image_cuda(image_name: str) -> str | None:
    """Extract CUDA version string from a Docker image tag."""
    m = _CUDA_DOTTED_RE.search(image_name) or _CUDA_COMPACT_RE.search(image_name)
    return f"{m.group(1)}.{m.group(2)}" if m else None


def _build_allowed_cuda_versions(
    cuda_min: str, cuda_max: str | None,
) -> list[str]:
    """Expand a CUDA min/max range into an explicit version list for RunPod."""
    min_major, min_minor = _parse_cuda_version(cuda_min)
    max_major, max_minor = _parse_cuda_version(cuda_max) if cuda_max else (99, 9)
    return [
        f"{major}.{minor}"
        for major in range(min_major, max_major + 1)
        for minor in range(
            min_minor if major == min_major else 0,
            (max_minor if major == max_major else 9) + 1,
        )
    ]


def _get_cuda_range(spec: PoolSpec) -> tuple[str | None, str | None]:
    """Extract CUDA min/max from the accelerator specification."""
    match spec.accelerator:
        case None:
            return None, None
        case str(name):
            catalog = SPECS.get(name)
            if catalog and "cuda" in catalog:
                return catalog["cuda"].get("min"), catalog["cuda"].get("max")
            return None, None
        case accel if accel.metadata and "cuda" in accel.metadata:
            cuda = accel.metadata["cuda"]
            return cuda.get("min"), cuda.get("max")
        case _:
            return None, None


async def _fetch_docker_tags() -> list[str]:
    """Fetch available image tags from Docker Hub (paginated)."""
    namespace, name = _DOCKER_HUB_REPO.split("/")
    all_tags: list[str] = []
    path = f"/v2/repositories/{namespace}/{name}/tags/"
    params: dict[str, str] | None = {"page_size": "100", "ordering": "-last_updated"}

    try:
        async with HttpClient(_DOCKER_HUB_URL, timeout=15) as http:
            for _ in range(5):
                data = await http.request("GET", path, params=params)
                if not data:
                    break
                all_tags.extend(tag["name"] for tag in data.get("results", []))
                next_url = data.get("next")
                if not next_url:
                    break
                path = next_url.removeprefix(_DOCKER_HUB_URL)
                params = None
    except Exception as e:
        log.warning("Failed to fetch Docker Hub tags: {err}", err=e)

    return all_tags


def _ubuntu_matches(tag: str, ubuntu: str) -> bool:
    if ubuntu == "newest":
        return True
    compact = ubuntu.replace(".", "")
    return f"ubuntu{compact}" in tag or f"ubuntu{ubuntu}" in tag


def _extract_ubuntu(tag: str) -> tuple[int, int]:
    m = _UBUNTU_RE.search(tag)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def _extract_tag_version(tag: str) -> tuple[int, int, int]:
    m = _TAG_VERSION_RE.match(tag)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else (0, 0, 0)


def _select_best_image(
    tags: list[str],
    cuda_min: tuple[int, int],
    cuda_max: tuple[int, int],
    ubuntu: str,
) -> str | None:
    """Select the newest image within the CUDA version range."""
    candidates: list[tuple[tuple[int, int, int], tuple[int, int], tuple[int, int], str]] = []
    for tag in tags:
        if tag.endswith("-test") or "-dev-" in tag or "ubuntu" not in tag:
            continue
        if not _ubuntu_matches(tag, ubuntu):
            continue
        m = _CUDA_DOTTED_RE.search(tag) or _CUDA_COMPACT_RE.search(tag)
        if not m:
            continue
        cuda_ver = (int(m.group(1)), int(m.group(2)))
        if cuda_min <= cuda_ver <= cuda_max:
            candidates.append((_extract_tag_version(tag), cuda_ver, _extract_ubuntu(tag), tag))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return f"{_DOCKER_HUB_REPO}:{candidates[0][3]}"


async def _resolve_image(
    spec: PoolSpec, config: RunPod, *, min_cuda: str | None = None,
) -> str:
    """Resolve the best RunPod container image for the given spec.

    Parameters
    ----------
    spec
        Pool specification with accelerator CUDA range.
    config
        RunPod provider config (ubuntu preference).
    min_cuda
        Minimum CUDA version with confirmed stock on RunPod hosts.
        When provided, overrides the catalog min so the image matches
        what machines actually support.
    """
    match getattr(spec.image, "container_image", None):
        case str() as img:
            return img
        case _:
            pass

    cuda_catalog_min, cuda_max = _get_cuda_range(spec)
    if cuda_catalog_min is None:
        return _FALLBACK_IMAGE

    cuda_min = min_cuda or cuda_catalog_min
    min_ver = _parse_cuda_version(cuda_min)
    max_ver = _parse_cuda_version(cuda_max) if cuda_max else (99, 99)

    tags = await _fetch_docker_tags()
    if image := _select_best_image(tags, min_ver, max_ver, config.ubuntu):
        log.info(
            "Selected image {image} for CUDA {min}-{max}",
            image=image, min=cuda_min, max=cuda_max,
        )
        return image

    log.warning(
        "No matching image for CUDA {min}-{max}, using fallback",
        min=cuda_min, max=cuda_max,
    )
    return _FALLBACK_IMAGE


@dataclass(frozen=True, slots=True)
class RunPodOfferData:
    """Carried in Offer.specific — includes probed CUDA availability."""

    gpu_type_id: str
    min_cuda: str | None


@dataclass(frozen=True, slots=True)
class RunPodSpecific:
    """RunPod-specific cluster data flowing through Cluster[RunPodSpecific]."""

    gpu_type_id: str | None
    cloud_type: str
    gpu_vram_gb: int = 0
    image_name: str = _FALLBACK_IMAGE
    registry_auth_id: str | None = None
    runpod_cluster_id: str | None = None
    pod_ids: tuple[tuple[int, str], ...] = ()
    cluster_ips: tuple[tuple[int, str], ...] = ()

    @property
    def is_instant_cluster(self) -> bool:
        return self.runpod_cluster_id is not None

    def pod_id_for(self, node_id: int) -> str | None:
        return next((pid for nid, pid in self.pod_ids if nid == node_id), None)

    def cluster_ip_for(self, node_id: int) -> str | None:
        return next((ip for nid, ip in self.cluster_ips if nid == node_id), None)


class RunPodProvider(Provider[RunPod, RunPodSpecific]):
    """Stateless RunPod provider. Holds only immutable config."""

    def __init__(self, config: RunPod) -> None:
        self._config = config

    @classmethod
    async def create(cls, config: RunPod) -> RunPodProvider:
        return cls(config)

    async def offers(self, spec: PoolSpec) -> AsyncIterator[Offer]:
        api_key = get_api_key(self._config.api_key)
        is_secure = self._config.cloud_type == CloudType.SECURE

        if not spec.accelerator_name:
            vcpus = spec.vcpus or 2
            memory_gb = spec.memory_gb or 4
            instance_id = f"cpu{self._config.cpu_clock}-{vcpus}-{memory_gb}"
            it = InstanceType(
                name=instance_id,
                accelerator=None,
                vcpus=float(vcpus),
                memory_gb=float(memory_gb),
                architecture="x86_64",
                specific=None,
            )
            yield Offer(
                id=f"runpod-cpu-{instance_id}",
                instance_type=it,
                spot_price=None,
                on_demand_price=None,
                billing_unit="hour",
                specific=None,
            )
            return

        cuda_min, cuda_max = _get_cuda_range(spec)
        if cuda_min and cuda_max:
            min_major = _parse_cuda_version(cuda_min)[0]
            max_major = _parse_cuda_version(cuda_max)[0]
            cuda_majors = [f"{m}.0" for m in range(max_major, min_major - 1, -1)]
        else:
            cuda_majors = [None]

        async with RunPodClient(api_key, config=self._config) as client:
            results = await asyncio.gather(*(
                client.get_gpu_types(
                    min_cuda_version=cv, secure_cloud=is_secure,
                )
                for cv in cuda_majors
            ))

        requested = spec.accelerator_name
        seen: set[str] = set()
        candidates: list[tuple[float, Offer]] = []

        for cuda_ver, gpu_types in zip(cuda_majors, results, strict=True):
            available = [
                g for g in gpu_types
                if (is_secure and g.get("secureCloud"))
                or (not is_secure and g.get("communityCloud"))
            ]

            for gpu in available:
                display = gpu.get("displayName", "")
                gpu_id = gpu.get("id", "")
                score = _gpu_match_score(requested, display, gpu_id)
                if score < _MIN_GPU_MATCH:
                    continue
                if gpu_id in seen:
                    continue

                lowest = gpu.get("lowestPrice") or {}
                spot_price = lowest.get("minimumBidPrice")
                on_demand_price = lowest.get("uninterruptablePrice")
                if spot_price is None and on_demand_price is None:
                    continue

                seen.add(gpu_id)
                vram_gb = gpu.get("memoryInGb", 0)
                gpu_count = spec.accelerator_count or 1
                accel = Accelerator(
                    name=display or gpu_id,
                    memory=f"{vram_gb}GB" if vram_gb else "",
                    count=gpu_count,
                )
                it = InstanceType(
                    name=gpu_id,
                    accelerator=accel,
                    vcpus=0.0,
                    memory_gb=0.0,
                    architecture="x86_64",
                    specific=None,
                )
                candidates.append((score, Offer(
                    id=f"runpod-{gpu_id}-cuda{cuda_ver or 'any'}",
                    instance_type=it,
                    spot_price=spot_price,
                    on_demand_price=on_demand_price,
                    billing_unit="hour",
                    specific=RunPodOfferData(gpu_id, cuda_ver),
                )))

        if not candidates:
            return

        best = max(s for s, _ in candidates)
        for score, offer in candidates:
            if score >= best - 1e-9:
                yield offer

    async def prepare(self, spec: PoolSpec, offer: Offer) -> Cluster[RunPodSpecific]:
        api_key = get_api_key(self._config.api_key)
        ssh_key_path = get_ssh_key_path()
        _, ssh_public_key = get_local_ssh_key()

        registry_auth_id: str | None = None
        async with RunPodClient(api_key, config=self._config) as client:
            await client.ensure_ssh_key(ssh_public_key)
            log.debug("SSH key ensured on RunPod account")

            if self._config.registry_auth:
                registry_auth_id = await client.resolve_registry_auth(self._config.registry_auth)
                if registry_auth_id:
                    log.info(
                        "Using registry credential {name!r}",
                        name=self._config.registry_auth,
                    )
                else:
                    log.warning(
                        "Registry credential {name!r} not found — pulls will be unauthenticated",
                        name=self._config.registry_auth,
                    )

        match offer.specific:
            case RunPodOfferData(gpu_type_id=gid, min_cuda=min_cuda):
                gpu_type_id: str | None = gid
            case _:
                gpu_type_id = None
                min_cuda = None

        gpu_vram_gb = 0
        if offer.instance_type.accelerator:
            mem = offer.instance_type.accelerator.memory
            gpu_vram_gb = int(mem.replace("GB", "")) if mem else 0

        image_name = await _resolve_image(spec, self._config, min_cuda=min_cuda)

        runpod_cluster_id: str | None = None
        pod_ids: tuple[tuple[int, str], ...] = ()
        cluster_ips: tuple[tuple[int, str], ...] = ()

        if self._config.cluster_mode == "instant" and spec.nodes >= 2 and gpu_type_id:
            runpod_cluster_id, pod_ids, cluster_ips = await _create_instant_cluster(
                self._config, spec, gpu_type_id, image_name,
                registry_auth_id=registry_auth_id,
            )

        shutdown_command = (
            "eval $(cat /proc/1/environ | tr '\\0' '\\n' "
            "| grep RUNPOD_ | sed 's/^/export /'); "
            "runpodctl remove pod $RUNPOD_POD_ID"
        )

        return Cluster(
            id=f"runpod-{uuid.uuid4().hex[:8]}",
            status="setting_up",
            spec=spec,
            offer=offer,
            ssh_key_path=ssh_key_path,
            ssh_user="root",
            use_sudo=False,
            shutdown_command=shutdown_command,
            specific=RunPodSpecific(
                gpu_type_id=gpu_type_id,
                cloud_type=self._config.cloud_type.value,
                gpu_vram_gb=gpu_vram_gb,
                image_name=image_name,
                registry_auth_id=registry_auth_id,
                runpod_cluster_id=runpod_cluster_id,
                pod_ids=pod_ids,
                cluster_ips=cluster_ips,
            ),
        )

    async def provision(
        self, cluster: Cluster[RunPodSpecific], count: int,
    ) -> tuple[Cluster[RunPodSpecific], Sequence[Instance]]:
        specific = cluster.specific
        api_key = get_api_key(self._config.api_key)

        if specific.is_instant_cluster:
            async with RunPodClient(api_key, config=self._config) as client:
                pods = await asyncio.gather(*(
                    client.get_pod(pid)
                    for _, pid in specific.pod_ids[:count]
                ))
            return cluster, [
                _build_runpod_instance(pod, "provisioning", cluster)
                if pod else Instance(id=pid, status="provisioning", offer=cluster.offer)
                for (_, pid), pod in zip(specific.pod_ids[:count], pods, strict=True)
            ]

        instances: list[Instance] = []
        async with RunPodClient(api_key, config=self._config) as client:
            for i in range(count):
                try:
                    pod = (
                        await _create_gpu_pod(client, self._config, cluster, i)
                        if specific.gpu_type_id
                        else await _create_cpu_pod(client, self._config, cluster)
                    )
                except RunPodError as e:
                    log.error("Failed to create pod: {err}", err=e)
                    continue
                instances.append(_build_runpod_instance(pod, "provisioning", cluster))

        return cluster, instances

    async def get_instance(
        self, cluster: Cluster[RunPodSpecific], instance_id: str,
    ) -> tuple[Cluster[RunPodSpecific], Instance | None]:
        api_key = get_api_key(self._config.api_key)
        async with RunPodClient(api_key, config=self._config) as client:
            pod = await client.get_pod(instance_id)

        if not pod:
            return cluster, None

        log.debug(
            "Pod {pid} status: desired={desired}, ip={ip}",
            pid=instance_id, desired=pod.get("desiredStatus"),
            ip=pod.get("publicIp"),
        )
        match pod.get("desiredStatus"):
            case "TERMINATED" | "EXITED":
                return cluster, None
            case "RUNNING" if pod.get("publicIp"):
                return cluster, _build_runpod_instance(pod, "provisioned", cluster)
            case _:
                return cluster, _build_runpod_instance(pod, "provisioning", cluster)

    async def terminate(
        self, cluster: Cluster[RunPodSpecific], instance_ids: tuple[str, ...],
    ) -> Cluster[RunPodSpecific]:
        if not instance_ids:
            return cluster

        api_key = get_api_key(self._config.api_key)
        async with RunPodClient(api_key, config=self._config) as client:
            async def _terminate(pod_id: str) -> None:
                try:
                    await client.terminate_pod(pod_id)
                except Exception as e:
                    log.error("Failed to terminate pod {pid}: {err}", pid=pod_id, err=e)

            await asyncio.gather(*(_terminate(pid) for pid in instance_ids))
        return cluster

    async def mount_endpoint(self, cluster: Cluster[RunPodSpecific]) -> MountEndpoint:
        from .client import get_api_key

        api_key = get_api_key(self._config.api_key)

        if self._config.data_center_ids == "global":
            raise ValueError(
                "RunPod volumes require an explicit data_center_ids setting. "
                f"S3-supported datacenters: {', '.join(sorted(_RUNPOD_S3_DATACENTERS))}"
            )

        dc = self._config.data_center_ids[0]
        if dc not in _RUNPOD_S3_DATACENTERS:
            raise ValueError(
                f"RunPod datacenter '{dc}' does not support S3 API. "
                f"Supported: {', '.join(sorted(_RUNPOD_S3_DATACENTERS))}"
            )

        dc_lower = dc.lower()
        return MountEndpoint(
            endpoint=f"https://s3api-{dc_lower}.runpod.io",
            access_key=api_key,
            secret_key=api_key,
        )

    async def teardown(self, cluster: Cluster[RunPodSpecific]) -> Cluster[RunPodSpecific]:
        specific = cluster.specific
        if not specific.is_instant_cluster:
            return cluster

        api_key = get_api_key(self._config.api_key)
        async with RunPodClient(api_key, config=self._config) as client:
            try:
                await client.delete_cluster(specific.runpod_cluster_id)  # type: ignore[arg-type]
            except Exception as e:
                log.error(
                    "Failed to delete cluster {cid}: {err}",
                    cid=specific.runpod_cluster_id, err=e,
                )
        return cluster


def _is_spot(allocation: str, offer: Offer) -> bool:
    match allocation:
        case "spot" | "spot-if-available":
            return offer.spot_price is not None
        case _:
            return False


def _build_runpod_instance(
    pod: PodResponse,
    status: InstanceStatus,
    cluster: Cluster[RunPodSpecific],
) -> Instance:
    specific = cluster.specific
    machine = pod.get("machine") or {}
    region = machine.get("dataCenterId") or machine.get("location") or ""

    private_ip: str | None = None
    if specific.is_instant_cluster:
        pod_id = pod["id"]
        node_id = next((nid for nid, pid in specific.pod_ids if pid == pod_id), None)
        if node_id is not None:
            private_ip = specific.cluster_ip_for(node_id)

    return Instance(
        id=pod["id"],
        status=status,
        offer=cluster.offer,
        ip=pod.get("publicIp") or "",
        private_ip=private_ip,
        ssh_port=get_ssh_port(pod),
        spot=_is_spot(cluster.spec.allocation, cluster.offer),
        region=region,
    )


async def _resolve_gpu_type(config: RunPod, spec: PoolSpec) -> tuple[str | None, int]:
    if not spec.accelerator_name:
        return None, 0

    api_key = get_api_key(config.api_key)
    async with RunPodClient(api_key, config=config) as client:
        gpu_types = await client.get_gpu_types()

    is_secure = config.cloud_type == CloudType.SECURE
    available = [
        g for g in gpu_types
        if (is_secure and g.get("secureCloud")) or (not is_secure and g.get("communityCloud"))
    ]
    log.debug(
        "GPU types: {total} total, {avail} available for cloud_type={cloud}",
        total=len(gpu_types), avail=len(available),
        cloud=config.cloud_type.value,
    )

    requested = spec.accelerator_name
    scored = [
        (_gpu_match_score(requested, g.get("displayName", ""), g.get("id", "")), g)
        for g in available
    ]
    scored = [(s, g) for s, g in scored if s >= _MIN_GPU_MATCH]

    if not scored:
        available_names = [g.get("displayName", g["id"]) for g in available]
        raise RuntimeError(
            f"No GPU type matches '{spec.accelerator_name}'. "
            f"Available: {', '.join(available_names)}"
        )

    scored.sort(key=lambda x: x[0], reverse=True)
    gpu = scored[0][1]
    log.info(
        "Selected GPU type {gpu_id} ({name})",
        gpu_id=gpu["id"], name=gpu.get("displayName"),
    )
    return gpu["id"], gpu.get("memoryInGb", 0)


async def _create_instant_cluster(
    config: RunPod,
    spec: PoolSpec,
    gpu_type_id: str,
    image_name: str,
    *,
    registry_auth_id: str | None = None,
) -> tuple[str, tuple[tuple[int, str], ...], tuple[tuple[int, str], ...]]:
    """Create a RunPod Instant Cluster.

    Returns
    -------
    tuple[str, tuple[tuple[int, str], ...], tuple[tuple[int, str], ...]]
        (cluster_id, pod_ids as (node_id, pod_id), cluster_ips as (node_id, ip))
    """
    log.info("Creating Instant Cluster with {n} nodes", n=spec.nodes)

    api_key = get_api_key(config.api_key)

    params: ClusterCreateParams = {
        "clusterName": f"skyward-{uuid.uuid4().hex[:8]}",
        "gpuTypeId": gpu_type_id,
        "podCount": spec.nodes,
        "gpuCountPerPod": spec.accelerator_count or 1,
        "type": "TRAINING",
        "imageName": image_name,
        "startSsh": True,
        "containerDiskInGb": config.container_disk_gb,
        "volumeInGb": config.volume_gb,
        "volumeMountPath": config.volume_mount_path,
        "ports": ",".join(config.ports),
        "deployCost": spec.max_hourly_cost or 10.0,
    }

    if config.data_center_ids != "global":
        params["dataCenterId"] = config.data_center_ids[0]
    if registry_auth_id:
        params["containerRegistryAuthId"] = registry_auth_id

    async with RunPodClient(api_key, config=config) as client:
        cluster_resp = await client.create_cluster(params)

    cluster_id = cluster_resp["id"]
    pod_ids: list[tuple[int, str]] = []
    cluster_ips: list[tuple[int, str]] = []

    for pod in cluster_resp["pods"]:
        node_id = pod.get("clusterIdx", len(pod_ids))
        pod_ids.append((node_id, pod["id"]))
        if cluster_ip := pod.get("clusterIp"):
            cluster_ips.append((node_id, cluster_ip))

    return cluster_id, tuple(pod_ids), tuple(cluster_ips)


async def _create_gpu_pod(
    client: RunPodClient,
    config: RunPod,
    cluster: Cluster[RunPodSpecific],
    node_index: int,
) -> PodResponse:
    use_spot = _is_spot(cluster.spec.allocation, cluster.offer)
    image_name = cluster.specific.image_name
    image_cuda = _extract_image_cuda(image_name)
    _, cuda_max = _get_cuda_range(cluster.spec)
    allowed_cuda = (
        _build_allowed_cuda_versions(image_cuda, cuda_max)
        if image_cuda else None
    )

    min_bid = cluster.offer.spot_price or 0.0
    bid = round(min_bid * config.bid_multiplier, 4) if use_spot else None

    log.debug(
        "Creating GPU pod for node {idx}: gpu={gpu}, count={count}, spot={spot}",
        idx=node_index, gpu=cluster.specific.gpu_type_id,
        count=cluster.spec.accelerator_count or 1, spot=use_spot,
    )

    return await client.deploy_gpu_pod(
        name=f"skyward-{cluster.id}-{node_index}",
        image_name=image_name,
        gpu_type_id=cluster.specific.gpu_type_id or "",
        gpu_count=cluster.spec.accelerator_count or 1,
        cloud_type=config.cloud_type.value.upper(),
        container_disk_gb=config.container_disk_gb,
        volume_gb=config.volume_gb,
        volume_mount_path=config.volume_mount_path,
        ports=",".join(config.ports),
        interruptible=use_spot,
        data_center_id=config.data_center_ids[0] if config.data_center_ids != "global" else None,
        deploy_cost=cluster.spec.max_hourly_cost,
        spot_price=bid,
        allowed_cuda_versions=allowed_cuda,
        container_registry_auth_id=cluster.specific.registry_auth_id,
    )


async def _create_cpu_pod(
    client: RunPodClient,
    config: RunPod,
    cluster: Cluster[RunPodSpecific],
) -> PodResponse:
    vcpus = cluster.spec.vcpus or 2
    memory_gb = cluster.spec.memory_gb or 4
    disk_gb = min(config.container_disk_gb, 20)
    instance_id = f"cpu{config.cpu_clock}-{vcpus}-{memory_gb}"

    params: CpuPodCreateParams = {
        "instanceId": instance_id,
        "cloudType": config.cloud_type.value.upper(),
        "containerDiskInGb": disk_gb,
        "startSsh": True,
        "templateId": "runpod-ubuntu",
        "ports": ",".join(config.ports),
        "deployCost": cluster.spec.max_hourly_cost or 0.50,
    }

    if config.data_center_ids != "global":
        params["dataCenterId"] = config.data_center_ids[0]
    if cluster.specific.registry_auth_id:
        params["containerRegistryAuthId"] = cluster.specific.registry_auth_id

    return await client.create_cpu_pod(params)
