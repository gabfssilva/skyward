"""RunPod Provider Actor - Casty behavior for RunPod GPU Pods.

Story: idle -> active -> stopped

idle: accepts ClusterRequested, registers SSH key, resolves GPU,
      optionally creates Instant Cluster, emits ClusterProvisioned.
active: handles InstanceRequested, BootstrapRequested, spawns
        instance_monitor for event streaming, and ShutdownRequested.
stopped: all pods terminated.
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from casty import ActorContext, ActorRef, Behavior, Behaviors
from loguru import logger

from skyward.actors.provider import BootstrapDone, ProviderMsg
from skyward.actors.streaming import instance_monitor
from skyward.messages import (
    BootstrapRequested,
    ClusterProvisioned,
    ClusterRequested,
    InstanceBootstrapped,
    InstanceLaunched,
    InstanceRequested,
    InstanceRunning,
    ShutdownRequested,
)
from skyward.providers.ssh_keys import get_local_ssh_key, get_ssh_key_path
from skyward.providers.wait import wait_for_ready

from .client import RunPodClient, RunPodError, get_api_key
from .config import CloudType, RunPod
from .state import RunPodClusterState
from .types import (
    ClusterCreateParams,
    CpuPodCreateParams,
    PodCreateParams,
    PodResponse,
    get_gpu_model,
    get_ssh_port,
)

if TYPE_CHECKING:
    from skyward.spec import PoolSpec


# =============================================================================
# Module-level helpers
# =============================================================================


def _get_image_name(spec: PoolSpec) -> str:
    if spec.image and hasattr(spec.image, "container_image"):
        return getattr(spec.image, "container_image")
    return "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"


async def _resolve_gpu_type(config: RunPod, spec: PoolSpec) -> str | None:
    if not spec.accelerator_name:
        return None

    api_key = get_api_key(config.api_key)
    async with RunPodClient(api_key) as client:
        gpu_types = await client.get_gpu_types()

    is_secure = config.cloud_type == CloudType.SECURE
    available = [
        g for g in gpu_types
        if (is_secure and g.get("secureCloud")) or (not is_secure and g.get("communityCloud"))
    ]

    requested = spec.accelerator_name.upper()
    for gpu in available:
        display_name = gpu.get("displayName", "").upper()
        gpu_id = gpu.get("id", "").upper()
        if requested in display_name or requested in gpu_id:
            logger.info(f"RunPod: Selected GPU type {gpu['id']} ({gpu.get('displayName')})")
            return gpu["id"]

    available_names = [g.get("displayName", g["id"]) for g in available]
    raise RuntimeError(
        f"No GPU type matches '{spec.accelerator_name}'. "
        f"Available: {', '.join(available_names)}"
    )


async def _create_gpu_pod(
    client: RunPodClient,
    config: RunPod,
    cluster: RunPodClusterState,
    event: InstanceRequested,
) -> PodResponse:
    use_spot = cluster.spec.allocation in ("spot", "spot-if-available")

    params: PodCreateParams = {
        "name": f"skyward-{cluster.cluster_id}-{event.node_id}",
        "imageName": _get_image_name(cluster.spec),
        "gpuTypeIds": [cluster.gpu_type_id or ""],
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
    cluster: RunPodClusterState,
) -> PodResponse:
    vcpus = cluster.spec.vcpus or 4
    memory_gb = cluster.spec.memory_gb or 8
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

    logger.info(f"RunPod: Creating CPU pod with instance {instance_id}")
    return await client.create_cpu_pod(params)


async def _create_instant_cluster(
    config: RunPod,
    state: RunPodClusterState,
    event: ClusterRequested,
) -> None:
    logger.info(f"RunPod: Creating Instant Cluster with {event.spec.nodes} nodes")

    api_key = get_api_key(config.api_key)
    cluster_type = "TRAINING"
    deploy_cost = event.spec.max_hourly_cost or 10.0

    params: ClusterCreateParams = {
        "clusterName": f"skyward-{state.cluster_id}",
        "gpuTypeId": state.gpu_type_id or "",
        "podCount": event.spec.nodes,
        "gpuCountPerPod": event.spec.accelerator_count or 1,
        "type": cluster_type,
        "imageName": _get_image_name(event.spec),
        "startSsh": True,
        "containerDiskInGb": config.container_disk_gb,
        "volumeInGb": config.volume_gb,
        "volumeMountPath": config.volume_mount_path,
        "ports": ",".join(config.ports),
        "deployCost": deploy_cost,
    }

    if config.data_center_ids != "global":
        params["dataCenterId"] = config.data_center_ids[0]

    try:
        async with RunPodClient(api_key) as client:
            cluster = await client.create_cluster(params)
    except RunPodError as e:
        logger.error(f"RunPod: Failed to create Instant Cluster: {e}")
        raise

    state.runpod_cluster_id = cluster["id"]
    logger.info(f"RunPod: Instant Cluster created with id {state.runpod_cluster_id}")

    for pod in cluster["pods"]:
        pod_id = pod["id"]
        node_id = pod.get("clusterIdx", len(state.pod_ids))
        state.pod_ids[node_id] = pod_id

        cluster_ip = pod.get("clusterIp")
        if cluster_ip:
            state.cluster_ips[node_id] = cluster_ip
            logger.debug(f"RunPod: Node {node_id} -> pod {pod_id}, cluster IP {cluster_ip}")
        else:
            logger.debug(f"RunPod: Node {node_id} -> pod {pod_id} (no cluster IP yet)")


async def _install_local_skyward(
    info: Any,
    cluster: RunPodClusterState,
) -> None:
    from skyward.providers.bootstrap import install_local_skyward, wait_for_ssh

    transport = await wait_for_ssh(
        host=info.ip,
        user=cluster.username,
        key_path=cluster.ssh_key_path,
        port=info.ssh_port,
        timeout=60.0,
        log_prefix="RunPod: ",
    )

    try:
        await install_local_skyward(
            transport=transport,
            info=info,
            log_prefix="RunPod: ",
            use_sudo=False,
        )
    finally:
        await transport.close()


async def _wait_for_running(
    config: RunPod,
    cluster: RunPodClusterState,
    event: InstanceLaunched,
    pool_ref: ActorRef,
) -> None:
    logger.info(f"RunPod: Waiting for pod {event.instance_id} to be running...")

    api_key = get_api_key(config.api_key)

    try:
        async with RunPodClient(api_key) as client:
            pod = await wait_for_ready(
                poll_fn=lambda: client.get_pod(event.instance_id),
                ready_check=lambda p: (
                    p is not None
                    and p.get("desiredStatus") == "RUNNING"
                    and bool(p.get("publicIp"))
                ),
                terminal_check=lambda p: (
                    p is not None and p.get("desiredStatus") == "TERMINATED"
                ),
                timeout=config.provision_timeout,
                interval=5.0,
                description=f"RunPod pod {event.instance_id}",
            )
    except TimeoutError:
        logger.error(f"RunPod: Pod {event.instance_id} did not become ready")
        return

    if not pod:
        logger.error(f"RunPod: Pod {event.instance_id} not found")
        return

    ip = pod.get("publicIp") or ""
    ssh_port = get_ssh_port(pod)
    hourly_rate = pod.get("costPerHr", 0.0)
    adjusted_rate = pod.get("adjustedCostPerHr", hourly_rate)

    private_ip: str | None = None
    if cluster.is_instant_cluster:
        private_ip = cluster.cluster_ips.get(event.node_id)
        if private_ip:
            logger.debug(f"RunPod: Using cluster IP {private_ip} for node {event.node_id}")

    cluster.hourly_rate = adjusted_rate
    cluster.on_demand_rate = hourly_rate
    cluster.gpu_count = pod.get("gpuCount", 0)
    cluster.gpu_model = cluster.spec.accelerator_name or get_gpu_model(pod)
    cluster.vcpus = pod.get("vcpuCount", 0)
    cluster.memory_gb = pod.get("memoryInGb", 0.0)

    pool_ref.tell(InstanceRunning(
        request_id=event.request_id,
        cluster_id=event.cluster_id,
        node_id=event.node_id,
        provider="runpod",
        instance_id=event.instance_id,
        ip=ip,
        private_ip=private_ip,
        ssh_port=ssh_port,
        spot=pod.get("interruptible", False),
        hourly_rate=adjusted_rate,
        on_demand_rate=hourly_rate,
        billing_increment=1,
        instance_type=cluster.gpu_type_id or "",
        gpu_count=cluster.gpu_count,
        gpu_model=cluster.gpu_model,
        vcpus=cluster.vcpus,
        memory_gb=cluster.memory_gb,
        gpu_vram_gb=cluster.gpu_vram_gb,
        region=cluster.region,
    ))


async def _run_bootstrap(
    config: RunPod,
    cluster: RunPodClusterState,
    event: BootstrapRequested,
) -> None:
    info = event.instance
    ttl = cluster.spec.ttl or config.instance_timeout
    bootstrap_script = cluster.spec.image.generate_bootstrap(
        ttl=ttl,
        shutdown_command="eval $(cat /proc/1/environ | tr '\\0' '\\n' | grep RUNPOD_ | sed 's/^/export /'); runpodctl remove pod $RUNPOD_POD_ID",
    )

    logger.info(f"RunPod: Connecting to {info.ip}:{info.ssh_port} to run bootstrap...")
    from skyward.providers.bootstrap import run_bootstrap_via_ssh, wait_for_ssh

    transport = await wait_for_ssh(
        host=info.ip,
        user=cluster.username,
        key_path=cluster.ssh_key_path,
        port=info.ssh_port,
        timeout=60.0,
    )

    try:
        await transport.run("mkdir -p /opt/skyward")
        await run_bootstrap_via_ssh(
            transport=transport,
            info=info,
            bootstrap_script=bootstrap_script,
            log_prefix="RunPod: ",
        )
    finally:
        await transport.close()

    logger.debug(f"RunPod: Bootstrap script launched on {info.id}")


# =============================================================================
# Actor behavior
# =============================================================================


def runpod_provider_actor(
    config: RunPod,
    pool_ref: ActorRef,
) -> Behavior[ProviderMsg]:

    def idle() -> Behavior[ProviderMsg]:
        async def receive(ctx: ActorContext[ProviderMsg], msg: ProviderMsg) -> Behavior[ProviderMsg]:
            match msg:
                case ClusterRequested(provider="runpod") as event:
                    cluster_id = f"runpod-{uuid.uuid4().hex[:8]}"

                    initial_region = (
                        config.data_center_ids[0]
                        if config.data_center_ids != "global"
                        else "global"
                    )
                    state = RunPodClusterState(
                        cluster_id=cluster_id,
                        spec=event.spec,
                        cloud_type=config.cloud_type.value,
                        data_center_ids=config.data_center_ids,
                        region=initial_region,
                    )

                    ssh_key_path = get_ssh_key_path()
                    _, ssh_public_key = get_local_ssh_key()
                    state.ssh_key_path = ssh_key_path
                    state.ssh_public_key = ssh_public_key

                    api_key = get_api_key(config.api_key)
                    async with RunPodClient(api_key) as client:
                        await client.ensure_ssh_key(ssh_public_key)
                        logger.debug("RunPod: SSH key registered on account")

                    gpu_type_id = await _resolve_gpu_type(config, event.spec)
                    state.gpu_type_id = gpu_type_id

                    if event.spec.nodes >= 2:
                        await _create_instant_cluster(config, state, event)

                    provisioned = ClusterProvisioned(
                        request_id=event.request_id,
                        cluster_id=cluster_id,
                        provider="runpod",
                    )
                    pool_ref.tell(provisioned)

                    return active(state)

                case _:
                    return Behaviors.same()

        return Behaviors.receive(receive)

    def active(state: RunPodClusterState) -> Behavior[ProviderMsg]:
        async def receive(ctx: ActorContext[ProviderMsg], msg: ProviderMsg) -> Behavior[ProviderMsg]:
            match msg:
                case InstanceRequested(provider="runpod") as event:
                    if event.cluster_id != state.cluster_id:
                        return Behaviors.same()

                    if state.is_instant_cluster:
                        pod_id = state.pod_ids.get(event.node_id)
                        if pod_id:
                            logger.info(f"RunPod: Instant Cluster pod {pod_id} for node {event.node_id}")
                            state.pending_nodes.add(event.node_id)
                            launched = InstanceLaunched(
                                request_id=event.request_id,
                                cluster_id=event.cluster_id,
                                node_id=event.node_id,
                                provider="runpod",
                                instance_id=pod_id,
                            )
                            asyncio.create_task(
                                _wait_for_running(config, state, launched, pool_ref)
                            )
                        return Behaviors.same()

                    logger.info(f"RunPod: Launching pod for node {event.node_id}")
                    api_key = get_api_key(config.api_key)

                    try:
                        async with RunPodClient(api_key) as client:
                            if state.gpu_type_id:
                                pod = await _create_gpu_pod(client, config, state, event)
                            else:
                                pod = await _create_cpu_pod(client, config, state)
                    except RunPodError as e:
                        logger.error(f"RunPod: Failed to create pod: {e}")
                        return Behaviors.same()

                    pod_id = pod["id"]
                    logger.debug(f"RunPod: Pod created with id {pod_id}")
                    state.pod_ids[event.node_id] = pod_id
                    state.pending_nodes.add(event.node_id)
                    machine = pod.get("machine") or {}
                    state.region = machine.get("dataCenterId") or machine.get("location") or ""

                    launched = InstanceLaunched(
                        request_id=event.request_id,
                        cluster_id=event.cluster_id,
                        node_id=event.node_id,
                        provider="runpod",
                        instance_id=pod_id,
                    )
                    asyncio.create_task(
                        _wait_for_running(config, state, launched, pool_ref)
                    )
                    return Behaviors.same()

                case BootstrapRequested(instance=info, cluster_id=cid) if info.provider == "runpod":
                    if cid != state.cluster_id:
                        return Behaviors.same()
                    asyncio.create_task(
                        _run_bootstrap(config, state, msg)
                    )
                    ctx.spawn(
                        instance_monitor(
                            info=info,
                            ssh_user=state.username,
                            ssh_key_path=state.ssh_key_path,
                            pool_ref=pool_ref,
                            reply_to=ctx.self,
                        ),
                        f"monitor-{info.id}",
                    )
                    return Behaviors.same()

                case BootstrapDone(instance=info, success=True) if info.provider == "runpod":
                    logger.info(f"RunPod: Bootstrap completed on {info.id}, skyward_source={state.spec.image.skyward_source}")
                    if state.spec.image and state.spec.image.skyward_source == "local":
                        logger.info(f"RunPod: Installing local skyward wheel on {info.id}...")
                        await _install_local_skyward(info, state)
                        logger.info(f"RunPod: Local skyward installed on {info.id}")
                    state.add_instance(info)
                    pool_ref.tell(InstanceBootstrapped(instance=info))
                    return Behaviors.same()

                case BootstrapDone(instance=info, success=False, error=error) if info.provider == "runpod":
                    logger.error(f"RunPod: Bootstrap failed on {info.id}: {error}")
                    return Behaviors.same()

                case ShutdownRequested() as event:
                    if event.cluster_id != state.cluster_id:
                        return Behaviors.same()

                    logger.info(f"RunPod: Shutting down cluster {state.cluster_id}")
                    api_key = get_api_key(config.api_key)

                    async with RunPodClient(api_key) as client:
                        if state.is_instant_cluster:
                            with suppress(Exception):
                                await client.delete_cluster(state.runpod_cluster_id)  # type: ignore[arg-type]
                        else:
                            for pod_id in state.pod_ids.values():
                                with suppress(Exception):
                                    await client.terminate_pod(pod_id)

                    pass
                    return Behaviors.stopped()

                case _:
                    return Behaviors.same()

        return Behaviors.receive(receive)

    return idle()


__all__ = ["runpod_provider_actor"]
