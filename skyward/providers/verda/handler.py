"""Verda Provider Actor - Casty behavior for Verda cloud lifecycle.

Story: idle -> active -> stopped.

The actor receives ProviderMsg and communicates lifecycle events
via pool_ref. Observability is provided transparently via Behaviors.spy().
"""

from __future__ import annotations

import re
import uuid
from contextlib import suppress
from typing import Any

from casty import ActorContext, ActorRef, Behavior, Behaviors
from loguru import logger

from skyward.actors.provider import BootstrapDone, ProviderMsg
from skyward.actors.streaming import instance_monitor
from skyward.messages import (
    BootstrapRequested,
    ClusterProvisioned,
    ClusterRequested,
    InstanceBootstrapped,
    InstanceRequested,
    InstanceRunning,
    ShutdownRequested,
)
from skyward.providers.ssh_keys import ensure_ssh_key_on_provider, get_ssh_key_path
from skyward.providers.wait import wait_for_ready
from skyward.spec import PoolSpec

from .client import VerdaClient, VerdaError
from .config import Verda
from .state import VerdaClusterState
from .types import (
    InstanceTypeResponse,
    get_accelerator,
    get_accelerator_count,
    get_accelerator_memory_gb,
    get_memory_gb,
    get_price_on_demand,
    get_price_spot,
    get_vcpu,
)


# =============================================================================
# Module-level helpers
# =============================================================================


async def _resolve_instance_type(
    client: VerdaClient, spec: PoolSpec
) -> tuple[str, str, InstanceTypeResponse]:
    use_spot = spec.allocation in ("spot", "spot-if-available")

    instance_types = await client.list_instance_types()
    availability = await client.get_availability(is_spot=use_spot)

    available_types = {t for region_types in availability.values() for t in region_types}

    def _matches(itype: InstanceTypeResponse) -> bool:
        if itype["instance_type"] not in available_types:
            return False
        if not spec.accelerator_name:
            return True
        accel = get_accelerator(itype)
        if not accel:
            return False
        accel_upper = accel.upper()
        requested_upper = spec.accelerator_name.upper()
        return accel_upper in requested_upper or requested_upper in accel_upper

    candidates = [itype for itype in instance_types if _matches(itype)]

    if not candidates:
        raise RuntimeError(f"No instance types match accelerator={spec.accelerator_name}")

    def sort_key(it: InstanceTypeResponse) -> float:
        price = get_price_spot(it) if use_spot else get_price_on_demand(it)
        return price if price is not None else float("inf")

    candidates.sort(key=sort_key)

    selected = candidates[0]
    supported_os = selected.get("supported_os", [])
    logger.debug(f"Verda: Instance {selected['instance_type']} supported_os: {supported_os}")

    os_image = _select_os_image(spec, supported_os)

    logger.debug(f"Verda: Selected {selected['instance_type']} with image {os_image}")
    return selected["instance_type"], os_image, selected


def _select_os_image(spec: PoolSpec, supported_os: list[str]) -> str:
    default_cuda_image = "ubuntu-22.04-cuda-12.1"

    if not spec.accelerator_name:
        return supported_os[0] if supported_os else "ubuntu-22.04"

    def is_preferred_image(img: str) -> bool:
        img_lower = img.lower()
        return (
            img_lower.startswith("ubuntu-")
            and "cuda" in img_lower
            and "kubernetes" not in img_lower
            and "jupyter" not in img_lower
            and "docker" not in img_lower
            and "cluster" not in img_lower
            and "open" not in img_lower
        )

    def parse_image_version(img: str) -> tuple[int, int, int, int]:
        ubuntu_match = re.search(r"ubuntu-(\d+)\.(\d+)", img.lower())
        cuda_match = re.search(r"cuda-?(\d+)\.(\d+)", img.lower())
        ubuntu_major = int(ubuntu_match.group(1)) if ubuntu_match else 0
        ubuntu_minor = int(ubuntu_match.group(2)) if ubuntu_match else 0
        cuda_major = int(cuda_match.group(1)) if cuda_match else 0
        cuda_minor = int(cuda_match.group(2)) if cuda_match else 0
        return (cuda_major, cuda_minor, ubuntu_major, ubuntu_minor)

    preferred = [os for os in supported_os if is_preferred_image(os)]
    if not preferred:
        preferred = [os for os in supported_os if os.lower().startswith("ubuntu-") and "cuda" in os.lower()]
    if not preferred:
        preferred = [os for os in supported_os if "cuda" in os.lower()]

    if preferred:
        preferred.sort(key=parse_image_version, reverse=True)
        return preferred[0]

    return supported_os[0] if supported_os else default_cuda_image


async def _find_available_region(
    client: VerdaClient, instance_type: str, is_spot: bool, preferred_region: str
) -> str:
    availability = await client.get_availability(is_spot)

    if preferred_region in availability and instance_type in availability[preferred_region]:
        return preferred_region

    for region, types in availability.items():
        if instance_type in types:
            logger.info(f"Verda: Auto-selected region {region}")
            return region

    raise RuntimeError(f"No region has instance type '{instance_type}' available")


def _generate_user_data(config: Verda, spec: PoolSpec) -> str:
    ttl = spec.ttl or config.instance_timeout
    return spec.image.generate_bootstrap(ttl=ttl)


async def _install_local_skyward(
    info: Any, cluster: VerdaClusterState
) -> None:
    from skyward.providers.bootstrap import install_local_skyward, wait_for_ssh

    ssh_key_path = get_ssh_key_path()

    transport = await wait_for_ssh(
        host=info.ip,
        user=cluster.username,
        key_path=ssh_key_path,
        timeout=60.0,
        log_prefix="Verda: ",
    )

    try:
        await install_local_skyward(
            transport=transport,
            info=info,
            log_prefix="Verda: ",
        )
    finally:
        await transport.close()


# =============================================================================
# Actor behavior
# =============================================================================


def verda_provider_actor(
    config: Verda,
    client: VerdaClient,
    pool_ref: ActorRef,
) -> Behavior[ProviderMsg]:
    """A Verda provider tells this story: idle -> active -> stopped."""

    def idle() -> Behavior[ProviderMsg]:
        async def receive(ctx: ActorContext[ProviderMsg], msg: ProviderMsg) -> Behavior[ProviderMsg]:
            match msg:
                case ClusterRequested(request_id=request_id, provider="verda", spec=spec):
                    logger.info(f"Verda: Provisioning cluster for {spec.nodes} nodes")

                    cluster_id = f"verda-{uuid.uuid4().hex[:8]}"

                    state = VerdaClusterState(
                        cluster_id=cluster_id,
                        spec=spec,
                        region=config.region,
                    )

                    ssh_key_id = await ensure_ssh_key_on_provider(
                        list_keys_fn=client.list_ssh_keys,
                        create_key_fn=lambda name, key: client.create_ssh_key(name, key),
                        provider_name="verda",
                    )
                    state.ssh_key_id = ssh_key_id
                    state.ssh_key_path = get_ssh_key_path()

                    instance_type, os_image, itype_data = await _resolve_instance_type(client, spec)
                    state.instance_type = instance_type
                    state.os_image = os_image

                    use_spot = spec.allocation in ("spot", "spot-if-available")
                    spot_price = get_price_spot(itype_data)
                    on_demand_price = get_price_on_demand(itype_data)
                    state.hourly_rate = (spot_price if use_spot and spot_price else on_demand_price) or 0.0
                    state.on_demand_rate = on_demand_price or 0.0
                    state.vcpus = get_vcpu(itype_data)
                    state.memory_gb = get_memory_gb(itype_data)
                    state.gpu_count = get_accelerator_count(itype_data)
                    state.gpu_model = get_accelerator(itype_data) or ""
                    state.gpu_vram_gb = int(get_accelerator_memory_gb(itype_data))

                    user_data = _generate_user_data(config, spec)
                    script_name = f"skyward-bootstrap-{cluster_id}"
                    script = await client.create_startup_script(script_name, user_data)
                    state.startup_script_id = script["id"]

                    event = ClusterProvisioned(
                        request_id=request_id,
                        cluster_id=cluster_id,
                        provider="verda",
                    )
                    pool_ref.tell(event)

                    return active(state)

                case _:
                    return Behaviors.same()

        return Behaviors.receive(receive)

    def active(
        state: VerdaClusterState,
    ) -> Behavior[ProviderMsg]:
        async def receive(ctx: ActorContext[ProviderMsg], msg: ProviderMsg) -> Behavior[ProviderMsg]:
            match msg:
                case InstanceRequested(
                    request_id=request_id,
                    cluster_id=cluster_id,
                    node_id=node_id,
                    provider="verda",
                ):
                    if not state.instance_type or not state.os_image:
                        return Behaviors.same()

                    logger.info(f"Verda: Launching instance for node {node_id}")

                    use_spot = state.spec.allocation in ("spot", "spot-if-available")

                    actual_region = await _find_available_region(
                        client, state.instance_type, use_spot, state.region
                    )

                    hostname = f"skyward-{state.cluster_id}-{node_id}"

                    try:
                        instance = await client.create_instance(
                            instance_type=state.instance_type,
                            image=state.os_image,
                            ssh_key_ids=[state.ssh_key_id] if state.ssh_key_id else [],
                            location=actual_region,
                            hostname=hostname,
                            description=f"Skyward managed - cluster {state.cluster_id}",
                            startup_script_id=state.startup_script_id,
                            is_spot=use_spot,
                        )
                    except VerdaError as e:
                        logger.error(f"Verda: Failed to create instance: {e}")
                        return Behaviors.same()

                    state.pending_nodes.add(node_id)

                    instance_id = str(instance["id"])

                    try:
                        info = await wait_for_ready(
                            poll_fn=lambda: client.get_instance(instance_id),
                            ready_check=lambda i: i is not None and i["status"] == "running" and bool(i.get("ip")),
                            terminal_check=lambda i: i is not None and i["status"] in ("error", "discontinued", "deleted"),
                            timeout=300.0,
                            interval=5.0,
                            description=f"Verda instance {instance_id}",
                        )
                    except TimeoutError:
                        logger.error(f"Verda: Instance {instance_id} did not become ready")
                        return Behaviors.same()

                    if not info:
                        logger.error(f"Verda: Instance {instance_id} not found")
                        return Behaviors.same()

                    pool_ref.tell(InstanceRunning(
                        request_id=request_id,
                        cluster_id=cluster_id,
                        node_id=node_id,
                        provider="verda",
                        instance_id=instance_id,
                        ip=info["ip"],
                        private_ip=info.get("private_ip"),
                        ssh_port=22,
                        spot=use_spot,
                        hourly_rate=state.hourly_rate,
                        on_demand_rate=state.on_demand_rate,
                        billing_increment=1,
                        instance_type=state.instance_type or "",
                        gpu_count=state.gpu_count,
                        gpu_model=state.gpu_model,
                        vcpus=state.vcpus,
                        memory_gb=state.memory_gb,
                        gpu_vram_gb=state.gpu_vram_gb,
                        region=state.region,
                    ))

                    return Behaviors.same()

                case BootstrapRequested(cluster_id=_, instance=instance_info) if instance_info.provider == "verda":
                    ctx.spawn(
                        instance_monitor(
                            info=instance_info,
                            ssh_user=state.username,
                            ssh_key_path=state.ssh_key_path,
                            pool_ref=pool_ref,
                            reply_to=ctx.self,
                        ),
                        f"monitor-{instance_info.id}",
                    )
                    return Behaviors.same()

                case BootstrapDone(instance=info, success=True):
                    if state.spec.image and state.spec.image.skyward_source == "local":
                        await _install_local_skyward(info, state)

                    state.add_instance(info)
                    pool_ref.tell(InstanceBootstrapped(instance=info))
                    return Behaviors.same()

                case BootstrapDone(instance=info, success=False, error=error):
                    logger.error(f"Verda: Bootstrap failed on {info.id}: {error}")
                    return Behaviors.same()

                case ShutdownRequested(cluster_id=cluster_id):
                    logger.info(f"Verda: Shutting down cluster {cluster_id}")

                    for instance_id in state.instance_ids:
                        with suppress(Exception):
                            await client.delete_instance(instance_id)

                    if state.startup_script_id:
                        with suppress(Exception):
                            await client.delete_startup_script(state.startup_script_id)

                    pass

                    return Behaviors.stopped()

                case _:
                    return Behaviors.same()

        return Behaviors.receive(receive)

    return idle()


__all__ = ["verda_provider_actor"]
