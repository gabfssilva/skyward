from __future__ import annotations

import asyncio
import re
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

import httpx

from skyward.accelerators import Accelerator
from skyward.core import PoolSpec
from skyward.core.model import Cluster, Instance, InstanceStatus, InstanceType, Offer
from skyward.observability.logger import logger
from skyward.providers.provider import Provider

if TYPE_CHECKING:
    from skyward.storage import Storage
from skyward.providers.ssh_keys import get_local_ssh_key, get_ssh_key_path

from .client import RunPodClient, RunPodError, get_api_key
from .config import RunPod
from .types import (
    ClusterCreateParams,
    CpuPodCreateParams,
    GpuTypeResponse,
    PodCreateParams,
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
_KNOWN_CUDA_VERSIONS = [
    "11.8", "12.0", "12.1", "12.2", "12.3", "12.4",
    "12.5", "12.6", "12.7", "12.8", "12.9", "13.0", "13.1",
]


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
        case Accelerator(metadata=metadata) if metadata and "cuda" in metadata:
            cuda = metadata["cuda"]
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
        async with httpx.AsyncClient(base_url=_DOCKER_HUB_URL, timeout=httpx.Timeout(15)) as http:
            for _ in range(5):
                resp = await http.get(path, params=params)
                if resp.status_code >= 400:
                    break
                data = resp.json()
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


def _select_image_candidates(
    tags: list[str],
    cuda_min: tuple[int, int],
    cuda_max: tuple[int, int],
    ubuntu: str,
) -> tuple[str, ...]:
    """Select best image per CUDA version within range, sorted highest first.

    For each distinct CUDA major.minor in the Docker Hub tags, picks the
    newest image tag. Returns them highest-CUDA-first so the deploy retry
    can walk the list top-down until a host accepts.
    """
    best_per_cuda: dict[
        tuple[int, int],
        tuple[tuple[int, int, int], tuple[int, int], str],
    ] = {}
    for tag in tags:
        if tag.endswith("-test") or "-dev-" in tag or "ubuntu" not in tag:
            continue
        if not _ubuntu_matches(tag, ubuntu):
            continue
        m = _CUDA_DOTTED_RE.search(tag) or _CUDA_COMPACT_RE.search(tag)
        if not m:
            continue
        cuda_ver = (int(m.group(1)), int(m.group(2)))
        if not (cuda_min <= cuda_ver <= cuda_max):
            continue
        tag_ver = _extract_tag_version(tag)
        ubuntu_ver = _extract_ubuntu(tag)
        prev = best_per_cuda.get(cuda_ver)
        if not prev or (tag_ver, ubuntu_ver) > (prev[0], prev[1]):
            best_per_cuda[cuda_ver] = (tag_ver, ubuntu_ver, tag)

    return tuple(
        f"{_DOCKER_HUB_REPO}:{entry[2]}"
        for _, entry in sorted(best_per_cuda.items(), reverse=True)
    )


async def _resolve_image_candidates(
    spec: PoolSpec, config: RunPod,
) -> tuple[str, ...]:
    """Resolve candidate RunPod container images, sorted highest CUDA first.

    Parameters
    ----------
    spec
        Pool specification with accelerator CUDA range.
    config
        RunPod provider config (ubuntu preference).
    """
    if config.container_image:
        return (config.container_image,)

    cuda_catalog_min, cuda_catalog_max = _get_cuda_range(spec)
    if cuda_catalog_min is None:
        return (_FALLBACK_IMAGE,)

    min_ver = _parse_cuda_version(cuda_catalog_min)
    max_ver = _parse_cuda_version(cuda_catalog_max) if cuda_catalog_max else (99, 99)

    tags = await _fetch_docker_tags()
    if candidates := _select_image_candidates(tags, min_ver, max_ver, config.ubuntu):
        log.info(
            "Resolved {n} image candidates for CUDA {min}-{max}",
            n=len(candidates), min=cuda_catalog_min, max=cuda_catalog_max,
        )
        return candidates

    log.warning(
        "No matching images for CUDA {min}-{max}, using fallback",
        min=cuda_catalog_min, max=cuda_catalog_max,
    )
    return (_FALLBACK_IMAGE,)


@dataclass(frozen=True, slots=True)
class RunPodOfferData:
    """Carried in Offer.specific — includes probed CUDA availability."""

    gpu_type_id: str
    proven_cuda: str | None
    cloud_type: str = "secure"


@dataclass(frozen=True, slots=True)
class RunPodSpecific:
    """RunPod-specific cluster data flowing through Cluster[RunPodSpecific]."""

    gpu_type_id: str | None
    cloud_type: str
    gpu_vram_gb: int = 0
    image_name: str = _FALLBACK_IMAGE
    image_candidates: tuple[str, ...] = ()
    registry_auth_id: str | None = None
    global_networking: bool = False
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

    name = "runpod"

    def __init__(self, config: RunPod) -> None:
        self._config = config

    @classmethod
    async def create(cls, config: RunPod) -> RunPodProvider:
        return cls(config)

    async def _fetch_gpu_types(self) -> list[tuple[GpuTypeResponse, str, str]]:
        """Fetch and deduplicate GPU types across clouds and CUDA probes.

        Returns
        -------
        list[tuple[GpuTypeResponse, str, str]]
            Triples of (gpu_data, cloud_type, proven_cuda_version).
        """
        api_key = get_api_key(self._config.api_key)
        cuda_probes = list(reversed(_KNOWN_CUDA_VERSIONS))

        async with RunPodClient(api_key, config=self._config) as client:
            all_results: list[tuple[str, list[list[GpuTypeResponse]]]] = []
            for cloud in ("community", "secure"):
                results = await asyncio.gather(*(
                    client.get_gpu_types(
                        min_cuda_version=cv,
                        secure_cloud=(cloud == "secure"),
                    )
                    for cv in cuda_probes
                ))
                all_results.append((cloud, results))

        seen: set[tuple[str, str]] = set()
        triples: list[tuple[GpuTypeResponse, str, str]] = []

        for cloud, results in all_results:
            is_secure = cloud == "secure"
            for cuda_ver, gpu_types in zip(cuda_probes, results, strict=True):
                available = [
                    g for g in gpu_types
                    if (is_secure and g.get("secureCloud"))
                    or (not is_secure and g.get("communityCloud"))
                ]
                for gpu in available:
                    gpu_id = gpu.get("id", "")
                    if (gpu_id, cloud) in seen:
                        continue

                    lowest = gpu.get("lowestPrice") or {}
                    if lowest.get("minimumBidPrice") is None and lowest.get("uninterruptablePrice") is None:
                        continue

                    seen.add((gpu_id, cloud))
                    triples.append((gpu, cloud, cuda_ver))

        return triples

    async def offers(self) -> AsyncIterator[Offer]:
        triples = await self._fetch_gpu_types()

        for gpu, cloud, cuda_ver in triples:
            display = gpu.get("displayName", "")
            gpu_id = gpu.get("id", "")
            vram_gb = gpu.get("memoryInGb", 0)
            lowest = gpu.get("lowestPrice") or {}
            base_spot = lowest.get("minimumBidPrice")
            base_on_demand = lowest.get("uninterruptablePrice")
            base_vcpu = float(lowest.get("minVcpu") or 0)
            base_memory = float(lowest.get("minMemory") or 0)
            is_secure = cloud == "secure"

            max_gpu = (
                gpu.get("maxGpuCountSecureCloud" if is_secure else "maxGpuCountCommunityCloud")
                or gpu.get("maxGpuCount")
                or 1
            )

            for gpu_count in range(1, max_gpu + 1):
                spot_price = round(base_spot * gpu_count, 4) if base_spot is not None else None
                on_demand_price = round(base_on_demand * gpu_count, 4) if base_on_demand is not None else None
                accel = Accelerator(
                    name=display or gpu_id,
                    memory=f"{vram_gb}GB" if vram_gb else "",
                    count=gpu_count,
                )
                it = InstanceType(
                    name=gpu_id,
                    accelerator=accel,
                    vcpus=base_vcpu * gpu_count,
                    memory_gb=base_memory * gpu_count,
                    architecture="x86_64",
                    specific=None,
                )
                yield Offer(
                    id=f"runpod-{gpu_id}-{cloud}-{gpu_count}x-cuda{cuda_ver or 'any'}",
                    instance_type=it,
                    spot_price=spot_price,
                    on_demand_price=on_demand_price,
                    billing_unit="hour",
                    specific=RunPodOfferData(gpu_id, cuda_ver, cloud),
                )

    def offer_filters(self) -> dict[str, str]:
        return {"cloud_type": self._config.cloud_type}

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
            case RunPodOfferData(gpu_type_id=gid):
                gpu_type_id: str | None = gid
            case _:
                gpu_type_id = None

        gpu_vram_gb = 0
        if offer.instance_type.accelerator:
            mem = offer.instance_type.accelerator.memory
            gpu_vram_gb = int(mem.replace("GB", "")) if mem else 0

        image_candidates = await _resolve_image_candidates(spec, self._config)
        image_name = image_candidates[0]

        match self._config.global_networking:
            case True:
                effective_global_networking = True
            case False:
                effective_global_networking = False
            case None:
                effective_global_networking = (
                    self._config.cluster_mode == "individual"
                    and spec.nodes.min >= 2
                    and gpu_type_id is not None
                )

        runpod_cluster_id: str | None = None
        pod_ids: tuple[tuple[int, str], ...] = ()
        cluster_ips: tuple[tuple[int, str], ...] = ()

        if self._config.cluster_mode == "instant" and spec.nodes.min >= 2 and gpu_type_id:
            runpod_cluster_id, pod_ids, cluster_ips = await _create_instant_cluster(
                self._config, spec, gpu_type_id, image_name,
                registry_auth_id=registry_auth_id,
            )

        shutdown_command = (
            "eval $(cat /proc/1/environ | tr '\\0' '\\n' "
            "| grep RUNPOD_ | sed 's/^/export /'); "
            "runpodctl remove pod $RUNPOD_POD_ID"
        )

        effective_offer = offer
        if (
            self._config.bid_multiplier != 1
            and _is_spot(spec.allocation, offer)
            and offer.spot_price is not None
        ):
            effective_offer = replace(
                offer,
                spot_price=round(offer.spot_price * self._config.bid_multiplier, 4),
            )

        return Cluster(
            id=f"runpod-{uuid.uuid4().hex[:8]}",
            status="setting_up",
            spec=spec,
            offer=effective_offer,
            ssh_key_path=ssh_key_path,
            ssh_user="root",
            use_sudo=False,
            shutdown_command=shutdown_command,
            specific=RunPodSpecific(
                gpu_type_id=gpu_type_id,
                cloud_type=self._config.cloud_type,
                gpu_vram_gb=gpu_vram_gb,
                image_name=image_name,
                image_candidates=image_candidates,
                registry_auth_id=registry_auth_id,
                global_networking=effective_global_networking,
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
            case "TERMINATED":
                return cluster, None
            case "EXITED":
                return cluster, _build_runpod_instance(pod, "exited", cluster)
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

    async def storage(self, cluster: Cluster[RunPodSpecific]) -> Storage:
        from skyward.storage import Storage

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
        return Storage(
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
    elif specific.global_networking:
        private_ip = f"{pod['id']}.runpod.internal"

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

    is_secure = config.cloud_type == 'secure'
    available = [
        g for g in gpu_types
        if (is_secure and g.get("secureCloud")) or (not is_secure and g.get("communityCloud"))
    ]
    log.debug(
        "GPU types: {total} total, {avail} available for cloud_type={cloud}",
        total=len(gpu_types), avail=len(available),
        cloud=config.cloud_type,
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
    log.info("Creating Instant Cluster with {n} nodes", n=spec.nodes.min)

    api_key = get_api_key(config.api_key)

    params: ClusterCreateParams = {
        "clusterName": f"skyward-{uuid.uuid4().hex[:8]}",
        "gpuTypeId": gpu_type_id,
        "podCount": spec.nodes.min,
        "gpuCountPerPod": int(spec.accelerator_count or 1),
        "type": "TRAINING",
        "imageName": image_name,
        "startSsh": True,
        "containerDiskInGb": spec.disk_gb or config.container_disk_gb,
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
    specific = cluster.specific
    use_spot = _is_spot(cluster.spec.allocation, cluster.offer)
    _, cuda_max = _get_cuda_range(cluster.spec)

    gpu_count = int(cluster.spec.accelerator_count or 1)
    bid_per_gpu = (
        round(cluster.offer.spot_price / gpu_count, 4)
        if use_spot and cluster.offer.spot_price is not None
        else None
    )
    candidates = specific.image_candidates or (specific.image_name,)

    if specific.global_networking:
        return await _create_gpu_pod_rest(
            client, config, cluster, node_index,
            image_name=candidates[0], use_spot=use_spot,
        )

    last_error: RunPodError | None = None
    for image_name in candidates:
        image_cuda = _extract_image_cuda(image_name)
        allowed_cuda = (
            _build_allowed_cuda_versions(image_cuda, cuda_max)
            if image_cuda and cuda_max else None
        )

        log.debug(
            "Trying pod for node {idx}: image={image}, cuda={cuda}, spot={spot}",
            idx=node_index, image=image_name, cuda=image_cuda, spot=use_spot,
        )

        try:
            return await client.deploy_gpu_pod(
                name=f"skyward-{cluster.id}-{node_index}",
                image_name=image_name,
                gpu_type_id=specific.gpu_type_id or "",
                gpu_count=int(cluster.spec.accelerator_count or 1),
                cloud_type=config.cloud_type.upper(),
                container_disk_gb=cluster.spec.disk_gb or config.container_disk_gb,
                volume_gb=config.volume_gb,
                volume_mount_path=config.volume_mount_path,
                ports=",".join(config.ports),
                interruptible=use_spot,
                data_center_id=config.data_center_ids[0] if config.data_center_ids != "global" else None,
                deploy_cost=cluster.spec.max_hourly_cost,
                spot_price=bid_per_gpu,
                allowed_cuda_versions=allowed_cuda,
                container_registry_auth_id=specific.registry_auth_id,
                min_download=int(config.min_inet_down) if config.min_inet_down is not None else None,
                min_upload=int(config.min_inet_up) if config.min_inet_up is not None else None,
            )
        except RunPodError as e:
            if "SUPPLY_CONSTRAINT" not in str(e):
                raise
            log.info(
                "No hosts for CUDA {cuda}, trying next candidate",
                cuda=image_cuda,
            )
            last_error = e

    raise last_error or RunPodError("All image candidates exhausted")


async def _create_gpu_pod_rest(
    client: RunPodClient,
    config: RunPod,
    cluster: Cluster[RunPodSpecific],
    node_index: int,
    *,
    image_name: str,
    use_spot: bool,
) -> PodResponse:
    params: PodCreateParams = {
        "name": f"skyward-{cluster.id}-{node_index}",
        "imageName": image_name,
        "gpuTypeIds": [cluster.specific.gpu_type_id or ""],
        "gpuCount": int(cluster.spec.accelerator_count or 1),
        "cloudType": config.cloud_type.upper(),
        "containerDiskInGb": cluster.spec.disk_gb or config.container_disk_gb,
        "volumeInGb": config.volume_gb,
        "volumeMountPath": config.volume_mount_path,
        "ports": list(config.ports),
        "interruptible": use_spot,
        "globalNetworking": True,
        "supportPublicIp": True,
    }
    if config.data_center_ids != "global":
        params["dataCenterIds"] = list(config.data_center_ids)
    if config.min_inet_down is not None:
        params["minDownload"] = int(config.min_inet_down)
    if config.min_inet_up is not None:
        params["minUpload"] = int(config.min_inet_up)
    return await client.create_pod(params)


async def _create_cpu_pod(
    client: RunPodClient,
    config: RunPod,
    cluster: Cluster[RunPodSpecific],
) -> PodResponse:
    vcpus = cluster.spec.vcpus or 2
    memory_gb = cluster.spec.memory_gb or 4
    disk_gb = min(cluster.spec.disk_gb or config.container_disk_gb, 20)
    instance_id = f"cpu{config.cpu_clock}-{vcpus}-{memory_gb}"

    params: CpuPodCreateParams = {
        "instanceId": instance_id,
        "cloudType": config.cloud_type.upper(),
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
