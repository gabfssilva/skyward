"""Vast.ai Provider Actor - Casty behavior for VastAI lifecycle.

Story: idle → active → stopped

idle: accepts ClusterRequested, provisions infra, transitions to active
active: handles InstanceRequested, BootstrapRequested, ShutdownRequested
stopped: actor terminates after shutdown

Note: VastAI has special handling:
- Overlay network for multi-node clusters
- Bootstrap via SSH (onstart_cmd has 1024 char limit)
- Offer reservation for parallel provisioning
"""

from __future__ import annotations

import asyncio
import random
import string
import uuid
from contextlib import suppress
from dataclasses import replace
from types import MappingProxyType
from typing import Any

from casty import ActorContext, ActorRef, Behavior, Behaviors
from loguru import logger

from skyward.actors.provider import BootstrapDone, ProviderMsg, _ProvisioningDone
from skyward.actors.streaming import instance_monitor
from skyward.messages import (
    BootstrapRequested,
    ClusterProvisioned,
    ClusterRequested,
    InstanceBootstrapped,
    InstanceRunning,
    InstanceRequested,
    ShutdownRequested,
)
from skyward.providers.ssh_keys import get_ssh_key_path
from skyward.providers.wait import wait_for_ready

from .client import VastAIClient, VastAIError, select_all_valid_clusters
from .config import VastAI
from .state import InstancePricing, VastAIClusterState
from .types import OfferResponse, get_direct_ssh_port


# =============================================================================
# Module-Level Helpers
# =============================================================================


async def _search_offers(
    client: VastAIClient,
    config: VastAI,
    state: VastAIClusterState,
) -> list[OfferResponse]:
    """Search for GPU offers matching cluster spec."""
    spec = state.spec
    use_interruptible = spec.allocation in ("spot", "spot-if-available")
    gpu_name = spec.accelerator_name.replace(" ", "_").replace("-", "_") if spec.accelerator_name else None

    offers = await client.search_offers(
        gpu_name=gpu_name,
        min_reliability=config.min_reliability,
        geolocation=config.geolocation,
        use_interruptible=use_interruptible,
        with_cluster_id=spec.nodes > 1,
    )

    logger.debug(f"VastAI: Got {len(offers)} offers from API for gpu_name={gpu_name}")
    if offers:
        unique_gpus = set(o["gpu_name"] for o in offers)
        logger.debug(f"VastAI: Available GPU types: {unique_gpus}")

    if gpu_name:
        req_norm = gpu_name.upper()
        filtered = [
            o for o in offers
            if req_norm in o["gpu_name"].replace(" ", "_").upper()
        ]
        logger.debug(f"VastAI: Filtered to {len(filtered)} offers matching '{req_norm}'")
        offers = filtered

    if state.overlay_cluster_id is not None:
        before_cluster_filter = len(offers)
        offers = [o for o in offers if o.get("cluster_id") == state.overlay_cluster_id]
        logger.debug(
            f"VastAI: Cluster filter: {len(offers)}/{before_cluster_filter} offers "
            f"in overlay cluster {state.overlay_cluster_id}"
        )
        if not offers:
            logger.error(
                f"VastAI: No offers in overlay cluster {state.overlay_cluster_id}. "
                f"Cluster may have become unavailable."
            )

    price_key = "min_bid" if use_interruptible else "dph_total"
    offers.sort(key=lambda o: o.get(price_key, float("inf")))

    if spec.max_hourly_cost:
        max_per_instance = spec.max_hourly_cost / spec.nodes

        def offer_price(o: OfferResponse) -> float:
            if use_interruptible:
                return o.get("min_bid", float("inf")) * config.bid_multiplier
            return o.get("dph_total", float("inf"))

        before_filter = len(offers)
        offers = [o for o in offers if offer_price(o) <= max_per_instance]

        if offers:
            logger.debug(
                f"VastAI: Budget filter: {len(offers)}/{before_filter} offers "
                f"within ${max_per_instance:.2f}/hr"
            )
        else:
            logger.warning(
                f"VastAI: No offers within budget ${max_per_instance:.2f}/hr "
                f"(filtered {before_filter} offers)"
            )

    return offers


async def _setup_overlay_network(
    client: VastAIClient,
    config: VastAI,
    state: VastAIClusterState,
) -> tuple[str | None, int | None]:
    """Set up overlay network for multi-node cluster.

    Returns (overlay_name, overlay_cluster_id) or (None, None) on failure.
    """
    spec = state.spec
    use_interruptible = spec.allocation in ("spot", "spot-if-available")
    gpu_name = spec.accelerator_name.replace(" ", "_").replace("-", "_") if spec.accelerator_name else None

    offers = await client.search_offers(
        gpu_name=gpu_name,
        min_reliability=config.min_reliability,
        geolocation=config.geolocation,
        use_interruptible=use_interruptible,
        with_cluster_id=True,
    )

    valid_clusters = select_all_valid_clusters(offers, spec.nodes, use_interruptible)
    if not valid_clusters:
        logger.warning(f"VastAI: No clusters found with {spec.nodes} nodes")
        return None, None

    for idx, (physical_cluster_id, _) in enumerate(valid_clusters):
        suffix = "".join(random.choices(string.ascii_lowercase, k=8))
        overlay_name = f"skyward-{suffix}"

        logger.info(f"VastAI: Trying cluster {physical_cluster_id} ({idx + 1}/{len(valid_clusters)})")

        try:
            await client.create_overlay(physical_cluster_id, overlay_name)
            logger.info(f"VastAI: Overlay '{overlay_name}' created")
            return overlay_name, physical_cluster_id
        except VastAIError as e:
            logger.warning(f"VastAI: Overlay failed on cluster {physical_cluster_id}: {e}")

    logger.warning("VastAI: Failed to create overlay on any cluster")
    return None, None


async def _detect_container_ip(
    ssh_host: str,
    ssh_port: int,
    timeout: float = 60.0,
) -> str:
    """Detect container's internal IP via SSH."""
    from skyward.providers.bootstrap import wait_for_ssh

    key_path = get_ssh_key_path()
    transport = await wait_for_ssh(
        host=ssh_host,
        user="root",
        key_path=key_path,
        timeout=timeout,
        port=ssh_port,
        log_prefix="VastAI: ",
    )

    try:
        _, output, _ = await transport.run("hostname -I | awk '{print $1}'")
        ip = output.strip()

        if ip:
            logger.info(f"VastAI: Detected container IP {ip}")
            return ip

        raise RuntimeError(f"Could not detect container IP. Output: {output!r}")
    finally:
        await transport.close()


async def _detect_overlay_ip(
    ssh_host: str,
    ssh_port: int,
    timeout: float = 120.0,
    max_retries: int = 30,
) -> tuple[str, str]:
    """Detect overlay network IP (10.x.x.x) and interface via SSH."""
    from skyward.providers.bootstrap import wait_for_ssh

    key_path = get_ssh_key_path()
    transport = await wait_for_ssh(
        host=ssh_host,
        user="root",
        key_path=key_path,
        timeout=timeout,
        port=ssh_port,
        log_prefix="VastAI: ",
    )

    try:
        cmd = r"""
IFACE=$(awk 'NR>1 && substr($2,7,2)=="0A" {print $1; exit}' /proc/net/route)
IP=$(hostname -I | tr ' ' '\n' | grep '^10\.')
echo "$IFACE $IP"
"""
        output = ""
        for attempt in range(1, max_retries + 1):
            _, output, _ = await transport.run(cmd.strip())
            parts = output.strip().split()

            if len(parts) >= 2:
                iface, ip = parts[0], parts[1]
                logger.info(f"VastAI: Detected overlay IP {ip} on {iface}")
                return ip, iface

            if attempt < max_retries:
                logger.debug(
                    f"VastAI: Overlay IP not ready (attempt {attempt}/{max_retries}), "
                    f"retrying in 2s..."
                )
                await asyncio.sleep(2)

        raise RuntimeError(
            f"Could not detect overlay IP after {max_retries} attempts. "
            f"Last output: {output.strip()!r}"
        )
    finally:
        await transport.close()


async def _install_local_skyward(
    instance_info: Any,
    ssh_host: str,
    ssh_port: int,
) -> None:
    """Install local skyward wheel on a remote instance."""
    from skyward.providers.bootstrap import install_local_skyward, wait_for_ssh

    key_path = get_ssh_key_path()

    transport = await wait_for_ssh(
        host=ssh_host,
        user="root",
        key_path=key_path,
        timeout=60.0,
        port=ssh_port,
        log_prefix="VastAI: ",
    )

    try:
        await install_local_skyward(
            transport=transport,
            info=instance_info,
            log_prefix="VastAI: ",
            use_sudo=False,
        )
    finally:
        await transport.close()


# =============================================================================
# Actor Behavior
# =============================================================================


def vastai_provider_actor(
    config: VastAI,
    client: VastAIClient,
    pool_ref: ActorRef,
) -> Behavior[ProviderMsg]:
    """Vast.ai provider tells this story: idle -> active -> stopped."""

    def idle() -> Behavior[ProviderMsg]:
        async def receive(ctx: ActorContext[ProviderMsg], msg: ProviderMsg) -> Behavior[ProviderMsg]:
            match msg:
                case ClusterRequested(request_id=request_id, provider="vastai", spec=spec):
                    logger.info(f"VastAI: Provisioning cluster for {spec.nodes} nodes")

                    cluster_id = f"vastai-{uuid.uuid4().hex[:8]}"
                    state = VastAIClusterState(
                        cluster_id=cluster_id,
                        spec=spec,
                        geolocation=config.geolocation,
                    )

                    async def provision() -> _ProvisioningDone:
                        async with client:
                            ssh_key_id, ssh_public_key = await client.ensure_ssh_key()

                            overlay_name = None
                            overlay_cluster_id = None
                            if spec.nodes > 1 and config.use_overlay:
                                overlay_name, overlay_cluster_id = await _setup_overlay_network(client, config, state)

                        new_state = replace(state,
                            ssh_key_id=ssh_key_id,
                            ssh_public_key=ssh_public_key,
                            overlay_name=overlay_name,
                            overlay_cluster_id=overlay_cluster_id,
                        )

                        pool_ref.tell(ClusterProvisioned(
                            request_id=request_id,
                            cluster_id=cluster_id,
                            provider="vastai",
                        ))

                        return _ProvisioningDone(state=new_state)

                    ctx.pipe_to_self(provision())
                    return active(state)

                case _:
                    return Behaviors.same()
        return Behaviors.receive(receive)

    def active(
        state: VastAIClusterState,
        reserved_offers: frozenset[int] = frozenset(),
    ) -> Behavior[ProviderMsg]:
        async def receive(ctx: ActorContext[ProviderMsg], msg: ProviderMsg) -> Behavior[ProviderMsg]:
            match msg:
                case _ProvisioningDone(state=new_state):
                    return active(new_state, reserved_offers)

                case InstanceRequested(
                    request_id=request_id,
                    provider="vastai",
                    cluster_id=cluster_id,
                    node_id=node_id,
                ):
                    logger.info(f"VastAI: Launching instance for node {node_id}")

                    async def launch_instance() -> _ProvisioningDone:
                        use_interruptible = state.spec.allocation in ("spot", "spot-if-available")
                        docker_image = state.docker_image or config.docker_image or VastAI.ubuntu()
                        label = f"skyward-{state.cluster_id}-{node_id}"
                        minimal_onstart = "#!/bin/bash\nset -e\nmkdir -p /opt/skyward\ntail -f /dev/null\n"

                        instance_id: int | None = None
                        last_error: str | None = None
                        new_reserved = reserved_offers
                        new_pricing = dict(state.instance_pricing)

                        async with client:
                            offers = await _search_offers(client, config, state)

                            if not offers:
                                logger.error(f"VastAI: No offers found for node {node_id}")
                                return _ProvisioningDone(state=state)

                            for idx, offer in enumerate(offers):
                                offer_id = offer["id"]

                                if offer_id in new_reserved:
                                    logger.debug(f"VastAI: Offer {offer_id} already reserved, skipping...")
                                    continue

                                new_reserved = new_reserved | frozenset({offer_id})

                                price = offer["min_bid"] * config.bid_multiplier if use_interruptible else None
                                price_display = price if price else offer.get("dph_total", 0)

                                logger.info(
                                    f"VastAI: Trying offer {idx + 1}/{len(offers)}: "
                                    f"machine_id={offer.get('machine_id')}, price=${price_display:.3f}/hr"
                                )

                                try:
                                    instance_id = await client.create_instance(
                                        offer_id=offer_id,
                                        image=docker_image,
                                        disk=config.disk_gb,
                                        label=label,
                                        onstart_cmd=minimal_onstart,
                                        price=price,
                                    )
                                    on_demand_rate = offer.get("dph_total", 0.0)
                                    hourly_rate = price if price else on_demand_rate
                                    new_pricing[str(instance_id)] = InstancePricing(
                                        hourly_rate=hourly_rate,
                                        on_demand_rate=on_demand_rate,
                                        gpu_name=offer.get("gpu_name", ""),
                                        gpu_count=offer.get("num_gpus", 0),
                                    )
                                    break
                                except VastAIError as e:
                                    new_reserved = new_reserved - frozenset({offer_id})
                                    last_error = str(e)
                                    logger.warning(f"VastAI: Offer {idx + 1}/{len(offers)} failed: {e}")
                                    continue

                        if instance_id is None:
                            logger.error(
                                f"VastAI: All offers failed for node {node_id}. "
                                f"Last error: {last_error}"
                            )
                            return _ProvisioningDone(state=state)

                        new_state = replace(state,
                            pending_nodes=state.pending_nodes | {node_id},
                            instance_pricing=MappingProxyType(new_pricing),
                        )

                        await _wait_and_emit_running(
                            client, config, new_state, pool_ref,
                            request_id, cluster_id, node_id, instance_id,
                        )

                        return _ProvisioningDone(state=new_state)

                    ctx.pipe_to_self(launch_instance())
                    return active(state, reserved_offers)

                case BootstrapRequested(
                    request_id=_,
                    instance=instance,
                    cluster_id=_,
                ):
                    logger.debug(f"VastAI: Starting bootstrap for instance {instance.id}")

                    async def run_bootstrap() -> None:
                        from skyward.providers.bootstrap import run_bootstrap_via_ssh, wait_for_ssh

                        key_path = get_ssh_key_path()
                        ttl = state.spec.ttl or config.instance_timeout
                        bootstrap_script = state.spec.image.generate_bootstrap(
                            ttl=ttl,
                            shutdown_command=(
                                "eval $(cat /proc/1/environ | tr '\\0' '\\n' | "
                                "grep -E 'CONTAINER_ID|CONTAINER_API_KEY' | sed 's/^/export /'); "
                                "curl -s -X DELETE https://console.vast.ai/api/v0/instances/$CONTAINER_ID/ "
                                "-H \"Authorization: Bearer $CONTAINER_API_KEY\""
                            ),
                        )

                        transport = await wait_for_ssh(
                            host=instance.ip,
                            user="root",
                            key_path=key_path,
                            timeout=120.0,
                            port=instance.ssh_port,
                            log_prefix="VastAI: ",
                        )

                        try:
                            await run_bootstrap_via_ssh(
                                transport=transport,
                                info=instance,
                                bootstrap_script=bootstrap_script,
                                log_prefix="VastAI: ",
                            )
                        finally:
                            await transport.close()

                    ctx.pipe_to_self(run_bootstrap())

                    ctx.spawn(
                        instance_monitor(
                            info=instance,
                            ssh_user="root",
                            ssh_key_path=get_ssh_key_path(),
                            pool_ref=pool_ref,
                            reply_to=ctx.self,
                        ),
                        f"monitor-{instance.id}",
                    )
                    return Behaviors.same()

                case BootstrapDone(instance=info, success=True):
                    logger.info(f"VastAI: Bootstrap completed for instance {info.id}")

                    if state.spec.image.skyward_source == "local":
                        async def install_and_register() -> _ProvisioningDone:
                            await _install_local_skyward(info, info.ip, info.ssh_port)
                            new_state = replace(state,
                                instances=MappingProxyType({**state.instances, info.id: info}),
                                pending_nodes=state.pending_nodes - {info.node},
                            )
                            pool_ref.tell(InstanceBootstrapped(instance=info))
                            return _ProvisioningDone(state=new_state)

                        ctx.pipe_to_self(install_and_register())
                    else:
                        new_state = replace(state,
                            instances=MappingProxyType({**state.instances, info.id: info}),
                            pending_nodes=state.pending_nodes - {info.node},
                        )
                        pool_ref.tell(InstanceBootstrapped(instance=info))
                        return active(new_state, reserved_offers)

                    return Behaviors.same()

                case BootstrapDone(instance=info, success=False, error=error):
                    logger.error(f"VastAI: Bootstrap failed for instance {info.id}: {error}")
                    return Behaviors.same()

                case ShutdownRequested(cluster_id=cluster_id):
                    logger.info(f"VastAI: Shutting down cluster {cluster_id}")

                    async def shutdown() -> None:
                        async with client:
                            for iid in state.instances.keys():
                                with suppress(Exception):
                                    await client.destroy_instance(int(iid))

                            if state.overlay_name:
                                with suppress(Exception):
                                    await client.delete_overlay(state.overlay_name)

                        await client.close()

                    ctx.pipe_to_self(shutdown())
                    return Behaviors.stopped()

                case _:
                    return Behaviors.same()

        return Behaviors.receive(receive)

    return idle()


# =============================================================================
# Internal Helpers
# =============================================================================


async def _wait_and_emit_running(
    client: VastAIClient,
    config: VastAI,
    state: VastAIClusterState,
    pool_ref: ActorRef,
    request_id: str,
    cluster_id: str,
    node_id: int,
    instance_id: int,
) -> None:
    """Wait for instance to be running, join overlay, detect IPs, emit InstanceRunning."""
    use_interruptible = state.spec.allocation in ("spot", "spot-if-available")
    str_id = str(instance_id)

    async with client:
        try:
            info = await wait_for_ready(
                poll_fn=lambda: client.get_instance(instance_id),
                ready_check=lambda i: (
                    i is not None
                    and i["actual_status"] == "running"
                    and bool(i.get("ssh_host") or i.get("public_ipaddr"))
                ),
                terminal_check=lambda i: (
                    i is not None
                    and i["actual_status"] in ("exited", "error", "destroyed")
                ),
                timeout=300.0,
                interval=5.0,
                description=f"VastAI instance {str_id}",
            )
        except TimeoutError:
            logger.error(f"VastAI: Instance {str_id} did not become ready")
            return

        if not info:
            logger.error(f"VastAI: Instance {str_id} not found")
            return

        direct_port = get_direct_ssh_port(info)
        logger.debug(
            f"VastAI: Instance {instance_id} connection info: "
            f"public_ipaddr={info.get('public_ipaddr')!r}, direct_port={direct_port}, "
            f"ssh_host={info.get('ssh_host')!r}, ports={info.get('ports')!r}"
        )
        if info.get("public_ipaddr") and direct_port:
            ssh_host = info["public_ipaddr"]
            ssh_port = direct_port
            logger.info(f"VastAI: Using direct IP {ssh_host}:{ssh_port}")
        else:
            ssh_host = info["ssh_host"]
            ssh_port = info.get("ssh_port", 22)
            logger.warning(f"VastAI: Falling back to SSH proxy {ssh_host}:{ssh_port}")

        private_ip = ""
        network_interface = ""

        if state.overlay_name:
            try:
                await client.join_overlay(state.overlay_name, instance_id)
                logger.info(f"VastAI: Instance {instance_id} joined overlay '{state.overlay_name}'")

                private_ip, network_interface = await _detect_overlay_ip(ssh_host, ssh_port)
            except VastAIError as e:
                logger.error(f"VastAI: Failed to join overlay '{state.overlay_name}': {e}")
                with suppress(Exception):
                    await client.destroy_instance(instance_id)
                return
            except RuntimeError as e:
                logger.error(f"VastAI: Failed to detect overlay IP: {e}")
                with suppress(Exception):
                    await client.destroy_instance(instance_id)
                return
        else:
            try:
                private_ip = await _detect_container_ip(ssh_host, ssh_port)
            except RuntimeError as e:
                logger.warning(f"VastAI: Could not detect container IP: {e}")
                private_ip = ssh_host

    pricing = state.instance_pricing.get(str_id)
    if pricing:
        hourly_rate = pricing.hourly_rate
        on_demand_rate = pricing.on_demand_rate
        gpu_name = pricing.gpu_name
        gpu_count = pricing.gpu_count
    else:
        hourly_rate = info.get("dph_total", 0.0)
        on_demand_rate = hourly_rate
        gpu_name = info.get("gpu_name", "")
        gpu_count = info.get("num_gpus", 0)

    vcpus = int(info.get("cpu_cores_effective", 0))
    memory_gb = info.get("cpu_ram", 0) / 1024
    total_vram_mb = info.get("gpu_ram", 0)
    gpu_vram_gb = int(total_vram_mb / 1024 / gpu_count) if gpu_count else 0

    pool_ref.tell(InstanceRunning(
        request_id=request_id,
        cluster_id=cluster_id,
        node_id=node_id,
        provider="vastai",
        instance_id=str_id,
        ip=ssh_host,
        private_ip=private_ip,
        ssh_port=ssh_port,
        spot=use_interruptible,
        network_interface=network_interface,
        hourly_rate=hourly_rate,
        on_demand_rate=on_demand_rate,
        billing_increment=1,
        instance_type=gpu_name,
        gpu_count=gpu_count,
        gpu_model=gpu_name,
        vcpus=vcpus,
        memory_gb=memory_gb,
        gpu_vram_gb=gpu_vram_gb,
        region=config.geolocation or "Global",
    ))


__all__ = ["vastai_provider_actor"]
