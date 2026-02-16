"""Verda Provider Actor - Casty behavior for Verda cloud lifecycle.

Story: idle -> active -> stopped.

The actor receives ProviderMsg and responds via reply_to on each message.
"""

from __future__ import annotations

import asyncio
import re
import uuid

from casty import ActorContext, ActorRef, Behavior, Behaviors

from skyward.actors.messages import (
    BootstrapDone,
    BootstrapRequested,
    ClusterProvisioned,
    ClusterRequested,
    InstanceBootstrapped,
    InstanceRequested,
    InstanceRunning,
    ProviderMsg,
    ShutdownCompleted,
    ShutdownRequested,
    _UserCodeSyncDone,
    _UserCodeSyncFailed,
)
from skyward.api.spec import PoolSpec
from skyward.observability.logger import logger
from skyward.providers.ssh_keys import ensure_ssh_key_on_provider, get_ssh_key_path
from skyward.providers.wait import wait_for_ready

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
    log = logger.bind(provider="verda")
    log.debug(
        "Instance {itype} supported_os: {os}",
        itype=selected["instance_type"], os=supported_os,
    )

    os_image = _select_os_image(spec, supported_os)

    log.debug("Selected {itype} with image {img}", itype=selected["instance_type"], img=os_image)
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
        preferred = [
            os for os in supported_os
            if os.lower().startswith("ubuntu-")
            and "cuda" in os.lower()
        ]
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
            logger.bind(provider="verda").info("Auto-selected region {region}", region=region)
            return region

    raise RuntimeError(f"No region has instance type '{instance_type}' available")


def _generate_user_data(config: Verda, spec: PoolSpec) -> str:
    ttl = spec.ttl or config.instance_timeout
    return spec.image.generate_bootstrap(ttl=ttl)


# =============================================================================
# Actor behavior
# =============================================================================


def verda_provider_actor(
    config: Verda,
    client: VerdaClient,
) -> Behavior[ProviderMsg]:
    """A Verda provider tells this story: idle -> active -> stopped."""
    log = logger.bind(provider="verda")

    def idle() -> Behavior[ProviderMsg]:
        async def receive(
            ctx: ActorContext[ProviderMsg], msg: ProviderMsg,
        ) -> Behavior[ProviderMsg]:
            match msg:
                case ClusterRequested(
                    request_id=request_id,
                    provider="verda",
                    spec=spec,
                    reply_to=caller,
                ):
                    log.info("Provisioning cluster for {n} nodes", n=spec.nodes)

                    cluster_id = f"verda-{uuid.uuid4().hex[:8]}"

                    state = VerdaClusterState(
                        cluster_id=cluster_id,
                        spec=spec,
                        region=config.region,
                    )

                    ssh_key_id = await ensure_ssh_key_on_provider(
                        list_keys_fn=client.list_ssh_keys,  # type: ignore[reportArgumentType]
                        create_key_fn=lambda name, key: client.create_ssh_key(name, key),  # type: ignore[reportArgumentType]
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
                    state.hourly_rate = (
                        spot_price
                        if use_spot and spot_price
                        else on_demand_price
                    ) or 0.0
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

                    if caller:
                        caller.tell(ClusterProvisioned(
                            request_id=request_id,
                            cluster_id=cluster_id,
                            provider="verda",
                        ))

                    return active(state)

                case _:
                    return Behaviors.same()

        return Behaviors.receive(receive)

    def active(
        state: VerdaClusterState,
        node_refs: dict[int, ActorRef] | None = None,
    ) -> Behavior[ProviderMsg]:
        refs = node_refs or {}
        alog = log.bind(state="active")

        async def receive(
            ctx: ActorContext[ProviderMsg], msg: ProviderMsg,
        ) -> Behavior[ProviderMsg]:
            alog.debug("received: {msg}", msg=type(msg).__name__)
            match msg:
                case InstanceRequested(
                    request_id=request_id,
                    cluster_id=cluster_id,
                    node_id=node_id,
                    provider="verda",
                    reply_to=node_ref,
                ):
                    if not state.instance_type or not state.os_image:
                        return Behaviors.same()

                    alog.info("Launching instance for node {nid}", nid=node_id)
                    new_refs = {**refs}
                    if node_ref:
                        new_refs[node_id] = node_ref

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
                        alog.error("Failed to create instance: {err}", err=e)
                        return Behaviors.same()

                    instance_id = str(instance["id"])
                    state.pending_nodes.add(node_id)
                    state.launched_ids.add(instance_id)

                    try:
                        info = await wait_for_ready(
                            poll_fn=lambda: client.get_instance(instance_id),
                            ready_check=lambda i: (
                                i is not None
                                and i["status"] == "running"
                                and bool(i.get("ip"))
                            ),
                            terminal_check=lambda i: (
                                i is not None
                                and i["status"] in (
                                    "error", "discontinued", "deleted",
                                )
                            ),
                            timeout=300.0,
                            interval=5.0,
                            description=f"Verda instance {instance_id}",
                        )
                    except TimeoutError:
                        alog.error("Instance {iid} did not become ready", iid=instance_id)
                        return Behaviors.same()

                    if not info:
                        alog.error("Instance {iid} not found", iid=instance_id)
                        return Behaviors.same()

                    running = InstanceRunning(
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
                        ssh_user=state.username,
                        ssh_key_path=state.ssh_key_path,
                    )
                    target = new_refs.get(node_id)
                    if target:
                        target.tell(running)

                    return active(state, new_refs)

                case BootstrapRequested(
                    cluster_id=_, instance=instance_info,
                ) if instance_info.provider == "verda":
                    return Behaviors.same()

                case BootstrapDone(instance=info, success=True):
                    if state.spec.image and state.spec.image.includes:
                        ctx.pipe_to_self(
                            coro=_sync_user_code_verda(info, state),
                            mapper=lambda _, i=info: _UserCodeSyncDone(instance=i),
                            on_failure=lambda e, i=info: _UserCodeSyncFailed(
                                instance=i, error=str(e),
                            ),
                        )
                    elif (target := refs.get(info.node)):
                        target.tell(InstanceBootstrapped(instance=info))
                    return Behaviors.same()

                case BootstrapDone(success=False):
                    return Behaviors.same()

                case _UserCodeSyncDone(instance=info):
                    if (target := refs.get(info.node)):
                        target.tell(InstanceBootstrapped(instance=info))
                    return Behaviors.same()

                case _UserCodeSyncFailed(instance=info, error=err):
                    alog.error("User code sync failed on {iid}: {err}", iid=info.id, err=err)
                    return Behaviors.same()

                case ShutdownRequested(
                    cluster_id=cid, reply_to=reply_to,
                ) if cid == state.cluster_id:
                    alog.info("ShutdownRequested matched for cluster {cid}", cid=cid)

                    async def _do_shutdown() -> None:
                        async def _delete(iid: str) -> None:
                            try:
                                await client.delete_instance(iid)
                            except Exception as e:
                                alog.error("Failed to delete instance {iid}: {err}", iid=iid, err=e)

                        await asyncio.gather(*(_delete(iid) for iid in state.launched_ids))

                        if state.startup_script_id:
                            try:
                                await client.delete_startup_script(state.startup_script_id)
                            except Exception as e:
                                alog.error("Failed to delete startup script: {err}", err=e)

                    ctx.pipe_to_self(
                        coro=_do_shutdown(),
                        mapper=lambda _: ShutdownCompleted(cluster_id=state.cluster_id),
                    )
                    return stopping(reply_to)

                case ShutdownRequested(cluster_id=cid):
                    alog.debug(
                        "ShutdownRequested cluster_id mismatch: {rcid} != {scid}",
                        rcid=cid, scid=state.cluster_id,
                    )
                    return Behaviors.same()

                case _:
                    return Behaviors.same()

        return Behaviors.receive(receive)

    def stopping(reply_to: ActorRef | None) -> Behavior[ProviderMsg]:
        async def receive(
            _ctx: ActorContext[ProviderMsg], msg: ProviderMsg,
        ) -> Behavior[ProviderMsg]:
            match msg:
                case ShutdownCompleted() as completed:
                    if reply_to is not None:
                        reply_to.tell(completed)
                    return Behaviors.stopped()
            return Behaviors.same()
        return Behaviors.receive(receive)

    return idle()


async def _sync_user_code_verda(
    info: object,
    state: VerdaClusterState,
) -> None:
    from skyward.providers.bootstrap import sync_user_code

    await sync_user_code(
        host=info.ip,  # type: ignore[attr-defined]
        user=state.username,
        key_path=state.ssh_key_path,
        port=info.ssh_port,  # type: ignore[attr-defined]
        image=state.spec.image,
        use_sudo=True,
        ssh_timeout=state.spec.ssh_timeout,
        ssh_retry_interval=state.spec.ssh_retry_interval,
    )
