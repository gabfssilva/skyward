from __future__ import annotations

import asyncio
import re
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from skyward.accelerators import Accelerator
from skyward.accelerators.catalog import SPECS
from skyward.api import PoolSpec
from skyward.api.model import Cluster, Instance, InstanceStatus, InstanceType, Offer
from skyward.infra.http import HttpClient
from skyward.observability.logger import logger
from skyward.providers.provider import Provider
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
_DOCKER_HUB_URL = "https://hub.docker.com"
_DOCKER_HUB_REPO = "runpod/base"


def _parse_cuda_version(version: str) -> tuple[int, int]:
    major, minor, *_ = version.split(".")
    return (int(major), int(minor))


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


async def _resolve_image(spec: PoolSpec, config: RunPod) -> str:
    """Resolve the best RunPod container image for the given spec."""
    match getattr(spec.image, "container_image", None):
        case str() as img:
            return img
        case _:
            pass

    cuda_min, cuda_max = _get_cuda_range(spec)
    if cuda_min is None:
        return _FALLBACK_IMAGE

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
class RunPodSpecific:
    """RunPod-specific cluster data flowing through Cluster[RunPodSpecific]."""

    gpu_type_id: str | None
    cloud_type: str
    gpu_vram_gb: int = 0
    image_name: str = _FALLBACK_IMAGE
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
            memory_gb = spec.memory_gb or 2
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

        async with RunPodClient(api_key, config=self._config) as client:
            gpu_types = await client.get_gpu_types()

        available = [
            g for g in gpu_types
            if (is_secure and g.get("secureCloud")) or (not is_secure and g.get("communityCloud"))
        ]

        log.debug(
            "GPU types: {total} total, {avail} available (cloud_type={ct})",
            total=len(gpu_types), avail=len(available),
            ct="secure" if is_secure else "community",
        )

        requested = spec.accelerator_name.upper()
        log.debug("Looking for accelerator: {req}", req=requested)
        for gpu in available:
            display = gpu.get("displayName", "")
            gpu_id = gpu.get("id", "")
            if requested not in display.upper() and requested not in gpu_id.upper():
                continue

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

            lowest = gpu.get("lowestPrice") or {}
            spot_price = lowest.get("minimumBidPrice")
            on_demand_price = lowest.get("minPrice")

            yield Offer(
                id=f"runpod-{gpu_id}",
                instance_type=it,
                spot_price=spot_price,
                on_demand_price=on_demand_price,
                billing_unit="hour",
                specific=gpu_id,
            )

    async def prepare(self, spec: PoolSpec, offer: Offer) -> Cluster[RunPodSpecific]:
        api_key = get_api_key(self._config.api_key)
        ssh_key_path = get_ssh_key_path()
        _, ssh_public_key = get_local_ssh_key()

        async with RunPodClient(api_key, config=self._config) as client:
            await client.ensure_ssh_key(ssh_public_key)
            log.debug("SSH key ensured on RunPod account")

        gpu_type_id: str | None = offer.specific if isinstance(offer.specific, str) else None
        gpu_vram_gb = 0
        if offer.instance_type.accelerator:
            mem = offer.instance_type.accelerator.memory
            gpu_vram_gb = int(mem.replace("GB", "")) if mem else 0

        image_name = await _resolve_image(spec, self._config)

        runpod_cluster_id: str | None = None
        pod_ids: tuple[tuple[int, str], ...] = ()
        cluster_ips: tuple[tuple[int, str], ...] = ()

        if spec.nodes >= 2 and gpu_type_id:
            runpod_cluster_id, pod_ids, cluster_ips = await _create_instant_cluster(
                self._config, spec, gpu_type_id, image_name,
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
        spot=pod.get("interruptible", False),
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

    requested = spec.accelerator_name.upper()
    gpu = next(
        (
            g for g in available
            if requested in g.get("displayName", "").upper()
            or requested in g.get("id", "").upper()
        ),
        None,
    )

    if gpu is not None:
        log.info(
            "Selected GPU type {gpu_id} ({name})",
            gpu_id=gpu["id"], name=gpu.get("displayName"),
        )
        return gpu["id"], gpu.get("memoryInGb", 0)

    available_names = [g.get("displayName", g["id"]) for g in available]
    raise RuntimeError(
        f"No GPU type matches '{spec.accelerator_name}'. "
        f"Available: {', '.join(available_names)}"
    )


async def _create_instant_cluster(
    config: RunPod,
    spec: PoolSpec,
    gpu_type_id: str,
    image_name: str,
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


def _extract_cuda_from_image(image_name: str) -> str | None:
    """Extract CUDA version string from image name for allowedCudaVersions."""
    tag = image_name.split(":")[-1] if ":" in image_name else image_name
    if m := _CUDA_DOTTED_RE.search(tag):
        return f"{m.group(1)}.{m.group(2)}"
    if m := _CUDA_COMPACT_RE.search(tag):
        return f"{m.group(1)}.{m.group(2)}"
    return None


async def _create_gpu_pod(
    client: RunPodClient,
    config: RunPod,
    cluster: Cluster[RunPodSpecific],
    node_index: int,
) -> PodResponse:
    use_spot = cluster.spec.allocation in ("spot", "spot-if-available")
    image_name = cluster.specific.image_name
    cuda_version = _extract_cuda_from_image(image_name)

    log.debug(
        "Creating GPU pod for node {idx}: gpu={gpu}, count={count}, cuda={cuda}",
        idx=node_index, gpu=cluster.specific.gpu_type_id,
        count=cluster.spec.accelerator_count or 1, cuda=cuda_version,
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
        allowed_cuda_versions=[cuda_version] if cuda_version else None,
    )


async def _create_cpu_pod(
    client: RunPodClient,
    config: RunPod,
    cluster: Cluster[RunPodSpecific],
) -> PodResponse:
    vcpus = cluster.spec.vcpus or 2
    memory_gb = cluster.spec.memory_gb or 2
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

    return await client.create_cpu_pod(params)
