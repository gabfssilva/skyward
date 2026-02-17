from __future__ import annotations

import asyncio
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from skyward.api import PoolSpec
from skyward.api.model import Cluster, Instance, InstanceStatus
from skyward.observability.logger import logger
from skyward.providers.provider import CloudProvider
from skyward.providers.ssh_keys import get_local_ssh_key, get_ssh_key_path

from .client import RunPodClient, RunPodError, get_api_key
from .config import CloudType, RunPod
from .types import (
    ClusterCreateParams,
    CpuPodCreateParams,
    PodCreateParams,
    PodResponse,
    get_gpu_model,
    get_ssh_port,
)

log = logger.bind(provider="runpod")

_DEFAULT_IMAGE = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"


def _get_image_name(spec: PoolSpec) -> str:
    match getattr(spec.image, "container_image", None):
        case str() as img:
            return img
        case _:
            return _DEFAULT_IMAGE


@dataclass(frozen=True, slots=True)
class RunPodSpecific:
    """RunPod-specific cluster data flowing through Cluster[RunPodSpecific]."""

    gpu_type_id: str | None
    cloud_type: str
    gpu_vram_gb: int = 0
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


class RunPodCloudProvider(CloudProvider[RunPod, RunPodSpecific]):
    """Stateless RunPod provider. Holds only immutable config."""

    def __init__(self, config: RunPod) -> None:
        self._config = config

    @classmethod
    async def create(cls, config: RunPod) -> RunPodCloudProvider:
        return cls(config)

    async def prepare(self, spec: PoolSpec) -> Cluster[RunPodSpecific]:
        api_key = get_api_key(self._config.api_key)
        ssh_key_path = get_ssh_key_path()
        _, ssh_public_key = get_local_ssh_key()

        async with RunPodClient(api_key, config=self._config) as client:
            await client.ensure_ssh_key(ssh_public_key)
            log.debug("SSH key ensured on RunPod account")

        gpu_type_id, gpu_vram_gb = await _resolve_gpu_type(self._config, spec)

        runpod_cluster_id: str | None = None
        pod_ids: tuple[tuple[int, str], ...] = ()
        cluster_ips: tuple[tuple[int, str], ...] = ()

        if spec.nodes >= 2 and gpu_type_id:
            runpod_cluster_id, pod_ids, cluster_ips = await _create_instant_cluster(
                self._config, spec, gpu_type_id,
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
            ssh_key_path=ssh_key_path,
            ssh_user="root",
            use_sudo=False,
            shutdown_command=shutdown_command,
            instances=(),
            specific=RunPodSpecific(
                gpu_type_id=gpu_type_id,
                cloud_type=self._config.cloud_type.value,
                gpu_vram_gb=gpu_vram_gb,
                runpod_cluster_id=runpod_cluster_id,
                pod_ids=pod_ids,
                cluster_ips=cluster_ips,
            ),
        )

    async def provision(
        self, cluster: Cluster[RunPodSpecific], count: int,
    ) -> Sequence[Instance]:
        specific = cluster.specific
        api_key = get_api_key(self._config.api_key)

        if specific.is_instant_cluster:
            return [
                Instance(id=pid, status="provisioning")
                for _, pid in specific.pod_ids[:count]
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

                instances.append(Instance(id=pod["id"], status="provisioning"))

        return instances

    async def get_instance(
        self, cluster: Cluster[RunPodSpecific], instance_id: str,
    ) -> Instance | None:
        api_key = get_api_key(self._config.api_key)
        async with RunPodClient(api_key, config=self._config) as client:
            pod = await client.get_pod(instance_id)

        if not pod:
            return None

        log.debug(
            "Pod {pid} status: desired={desired}, ip={ip}",
            pid=instance_id, desired=pod.get("desiredStatus"),
            ip=pod.get("publicIp"),
        )
        match pod.get("desiredStatus"):
            case "TERMINATED":
                return None
            case "RUNNING" if pod.get("publicIp"):
                return _build_runpod_instance(pod, "provisioned", cluster.specific)
            case _:
                return _build_runpod_instance(pod, "provisioning", cluster.specific)

    async def terminate(self, instance_ids: tuple[str, ...]) -> None:
        if not instance_ids:
            return

        api_key = get_api_key(self._config.api_key)
        async with RunPodClient(api_key, config=self._config) as client:
            async def _terminate(pod_id: str) -> None:
                try:
                    await client.terminate_pod(pod_id)
                except Exception as e:
                    log.error("Failed to terminate pod {pid}: {err}", pid=pod_id, err=e)

            await asyncio.gather(*(_terminate(pid) for pid in instance_ids))

    async def teardown(self, cluster: Cluster[RunPodSpecific]) -> None:
        specific = cluster.specific
        if not specific.is_instant_cluster:
            return

        api_key = get_api_key(self._config.api_key)
        async with RunPodClient(api_key, config=self._config) as client:
            try:
                await client.delete_cluster(specific.runpod_cluster_id)  # type: ignore[arg-type]
            except Exception as e:
                log.error(
                    "Failed to delete cluster {cid}: {err}",
                    cid=specific.runpod_cluster_id, err=e,
                )


def _build_runpod_instance(
    pod: PodResponse, status: InstanceStatus, specific: RunPodSpecific,
) -> Instance:
    gpu = pod.get("gpu")
    instance_type = gpu.get("id", "") if gpu else ""
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
        ip=pod.get("publicIp") or "",
        private_ip=private_ip,
        ssh_port=get_ssh_port(pod),
        spot=pod.get("interruptible", False),
        instance_type=instance_type,
        gpu_count=pod.get("gpuCount", 0),
        gpu_model=get_gpu_model(pod),
        vcpus=pod.get("vcpuCount", 0),
        memory_gb=pod.get("memoryInGb", 0.0),
        region=region,
        gpu_vram_gb=specific.gpu_vram_gb,
        hourly_rate=pod.get("adjustedCostPerHr", 0.0),
        on_demand_rate=pod.get("costPerHr", 0.0),
        billing_increment=1,
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
) -> tuple[str, tuple[tuple[int, str], ...], tuple[tuple[int, str], ...]]:
    """Create a RunPod Instant Cluster.

    Returns
    -------
    tuple[str, tuple[tuple[int, str], ...], tuple[tuple[int, str], ...]]
        (cluster_id, pod_ids as (node_id, pod_id), cluster_ips as (node_id, ip))
    """
    log.info("Creating Instant Cluster with {n} nodes", n=spec.nodes)

    api_key = get_api_key(config.api_key)
    image_name = _get_image_name(spec)

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


async def _create_gpu_pod(
    client: RunPodClient,
    config: RunPod,
    cluster: Cluster[RunPodSpecific],
    node_index: int,
) -> PodResponse:
    use_spot = cluster.spec.allocation in ("spot", "spot-if-available")

    log.debug(
        "Creating GPU pod for node {idx}: gpu={gpu}, count={count}",
        idx=node_index, gpu=cluster.specific.gpu_type_id,
        count=cluster.spec.accelerator_count or 1,
    )
    params: PodCreateParams = {
        "name": f"skyward-{cluster.id}-{node_index}",
        "imageName": _get_image_name(cluster.spec),
        "gpuTypeIds": [cluster.specific.gpu_type_id or ""],
        "gpuCount": cluster.spec.accelerator_count or 1,
        "cloudType": config.cloud_type.value.upper(),
        "containerDiskInGb": config.container_disk_gb,
        "volumeInGb": config.volume_gb,
        "volumeMountPath": config.volume_mount_path,
        "ports": list(config.ports),
        "interruptible": use_spot,
    }

    if config.data_center_ids != "global":
        params["dataCenterIds"] = list(config.data_center_ids)

    return await client.create_pod(params)


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
