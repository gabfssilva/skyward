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
    _ProvisioningDone,
    _UserCodeSyncDone,
    _UserCodeSyncFailed,
)
from skyward.observability.logger import logger
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
    log = logger.bind(provider="vastai")
    spec = state.spec
    use_interruptible = spec.allocation in ("spot", "spot-if-available")
    gpu_name = (
        spec.accelerator_name.replace(" ", "_").replace("-", "_")
        if spec.accelerator_name
        else None
    )

    offers = await client.search_offers(
        gpu_name=gpu_name,
        min_reliability=config.min_reliability,
        geolocation=config.geolocation,
        use_interruptible=use_interruptible,
        with_cluster_id=spec.nodes > 1,
    )

    log.debug("Got {n} offers from API for gpu_name={gpu}", n=len(offers), gpu=gpu_name)
    if offers:
        unique_gpus = {o["gpu_name"] for o in offers}
        log.debug("Available GPU types: {gpus}", gpus=unique_gpus)

    if gpu_name:
        req_norm = gpu_name.upper()
        filtered = [
            o for o in offers
            if req_norm in o["gpu_name"].replace(" ", "_").upper()
        ]
        log.debug("Filtered to {n} offers matching '{req}'", n=len(filtered), req=req_norm)
        offers = filtered

    if state.overlay_cluster_id is not None:
        before_cluster_filter = len(offers)
        offers = [o for o in offers if o.get("cluster_id") == state.overlay_cluster_id]
        log.debug(
            "Cluster filter: {n}/{total} offers in overlay cluster {cid}",
            n=len(offers), total=before_cluster_filter, cid=state.overlay_cluster_id,
        )
        if not offers:
            log.error(
                "No offers in overlay cluster {cid}, cluster may have become unavailable",
                cid=state.overlay_cluster_id,
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
            log.debug(
                "Budget filter: {n}/{total} offers within ${budget:.2f}/hr",
                n=len(offers), total=before_filter, budget=max_per_instance,
            )
        else:
            log.warning(
                "No offers within budget ${budget:.2f}/hr (filtered {total} offers)",
                budget=max_per_instance, total=before_filter,
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
    gpu_name = (
        spec.accelerator_name.replace(" ", "_").replace("-", "_")
        if spec.accelerator_name
        else None
    )

    offers = await client.search_offers(
        gpu_name=gpu_name,
        min_reliability=config.min_reliability,
        geolocation=config.geolocation,
        use_interruptible=use_interruptible,
        with_cluster_id=True,
    )

    log = logger.bind(provider="vastai")
    valid_clusters = select_all_valid_clusters(offers, spec.nodes, use_interruptible)
    if not valid_clusters:
        log.warning("No clusters found with {n} nodes", n=spec.nodes)
        return None, None

    for idx, (physical_cluster_id, _) in enumerate(valid_clusters):
        suffix = "".join(random.choices(string.ascii_lowercase, k=8))
        overlay_name = f"skyward-{suffix}"

        log.info(
            "Trying cluster {cid} ({i}/{total})",
            cid=physical_cluster_id, i=idx + 1, total=len(valid_clusters),
        )

        try:
            await client.create_overlay(physical_cluster_id, overlay_name)
            log.info("Overlay '{name}' created", name=overlay_name)
            return overlay_name, physical_cluster_id
        except VastAIError as e:
            log.warning("Overlay failed on cluster {cid}: {err}", cid=physical_cluster_id, err=e)

    log.warning("Failed to create overlay on any cluster")
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
            logger.bind(provider="vastai").info("Detected container IP {ip}", ip=ip)
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
                logger.bind(provider="vastai").info(
                    "Detected overlay IP {ip} on {iface}",
                    ip=ip, iface=iface,
                )
                return ip, iface

            if attempt < max_retries:
                logger.bind(provider="vastai").debug(
                    "Overlay IP not ready (attempt {a}/{total}), retrying in 2s",
                    a=attempt, total=max_retries,
                )
                await asyncio.sleep(2)

        raise RuntimeError(
            f"Could not detect overlay IP after {max_retries} attempts. "
            f"Last output: {output.strip()!r}"
        )
    finally:
        await transport.close()


# =============================================================================
# Actor Behavior
# =============================================================================


def vastai_provider_actor(
    config: VastAI,
    client: VastAIClient,
) -> Behavior[ProviderMsg]:
    """Vast.ai provider tells this story: idle -> active -> stopped."""
    log = logger.bind(provider="vastai")

    def idle() -> Behavior[ProviderMsg]:
        async def receive(
            ctx: ActorContext[ProviderMsg], msg: ProviderMsg,
        ) -> Behavior[ProviderMsg]:
            match msg:
                case ClusterRequested(
                    request_id=request_id,
                    provider="vastai",
                    spec=spec,
                    reply_to=caller,
                ):
                    log.info("Provisioning cluster for {n} nodes", n=spec.nodes)

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
                                (
                                overlay_name,
                                overlay_cluster_id,
                            ) = await _setup_overlay_network(
                                client, config, state,
                            )

                        new_state = replace(state,
                            ssh_key_id=ssh_key_id,
                            ssh_public_key=ssh_public_key,
                            overlay_name=overlay_name,
                            overlay_cluster_id=overlay_cluster_id,
                        )

                        if caller:
                            caller.tell(ClusterProvisioned(
                                request_id=request_id,
                                cluster_id=cluster_id,
                                provider="vastai",
                            ))

                        return _ProvisioningDone(state=new_state)

                    ctx.pipe_to_self(provision(), mapper=lambda x: x)  # type: ignore[reportArgumentType]
                    return active(state)

                case _:
                    return Behaviors.same()
        return Behaviors.receive(receive)

    def active(
        state: VastAIClusterState,
        reserved_offers: frozenset[int] = frozenset(),
        node_refs: dict[int, ActorRef] | None = None,
    ) -> Behavior[ProviderMsg]:
        refs = node_refs or {}
        alog = log.bind(state="active")

        async def receive(
            ctx: ActorContext[ProviderMsg], msg: ProviderMsg,
        ) -> Behavior[ProviderMsg]:
            alog.debug("Received: {msg}", msg=type(msg).__name__)
            match msg:
                case _ProvisioningDone(state=new_state):
                    return active(new_state, reserved_offers, refs)

                case InstanceRequested(
                    request_id=request_id,
                    provider="vastai",
                    cluster_id=cluster_id,
                    node_id=node_id,
                    reply_to=node_ref,
                ):
                    alog.info("Launching instance for node {nid}", nid=node_id)
                    new_refs = {**refs}
                    if node_ref:
                        new_refs[node_id] = node_ref

                    async def launch_instance() -> _ProvisioningDone:
                        use_interruptible = state.spec.allocation in ("spot", "spot-if-available")
                        docker_image = (
                            state.docker_image
                            or config.docker_image
                            or VastAI.ubuntu()
                        )
                        label = f"skyward-{state.cluster_id}-{node_id}"
                        minimal_onstart = (
                            "#!/bin/bash\nset -e\n"
                            "mkdir -p /opt/skyward\n"
                            "tail -f /dev/null\n"
                        )

                        instance_id: int | None = None
                        last_error: str | None = None
                        new_reserved = reserved_offers
                        new_pricing = dict(state.instance_pricing)

                        async with client:
                            offers = await _search_offers(client, config, state)

                            if not offers:
                                alog.error("No offers found for node {nid}", nid=node_id)
                                return _ProvisioningDone(state=state)

                            for idx, offer in enumerate(offers):
                                offer_id = offer["id"]

                                if offer_id in new_reserved:
                                    alog.debug(
                                        "Offer {oid} already reserved, skipping",
                                        oid=offer_id,
                                    )
                                    continue

                                new_reserved = new_reserved | frozenset({offer_id})

                                price = (
                                    offer["min_bid"]
                                    * config.bid_multiplier
                                    if use_interruptible
                                    else None
                                )
                                price_display = (
                                    price
                                    if price
                                    else offer.get("dph_total", 0)
                                )

                                alog.info(
                                    "Trying offer {i}/{total}: "
                                    "machine_id={mid}, price=${price:.3f}/hr",
                                    i=idx + 1, total=len(offers),
                                    mid=offer.get("machine_id"),
                                    price=price_display,
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
                                    alog.warning(
                                        "Offer {i}/{total} failed: {err}",
                                        i=idx + 1, total=len(offers), err=e,
                                    )
                                    continue

                        if instance_id is None:
                            alog.error(
                                "All offers failed for node {nid}, last error: {err}",
                                nid=node_id, err=last_error,
                            )
                            return _ProvisioningDone(state=state)

                        new_state = replace(state,
                            pending_nodes=state.pending_nodes | {node_id},
                            instance_pricing=MappingProxyType(new_pricing),
                        )

                        await _wait_and_emit_running(
                            client, config, new_state, new_refs,
                            request_id, cluster_id, node_id, instance_id,
                        )

                        return _ProvisioningDone(state=new_state)

                    ctx.pipe_to_self(launch_instance(), mapper=lambda x: x)  # type: ignore[reportArgumentType]
                    return active(state, reserved_offers, new_refs)

                case BootstrapRequested(
                    request_id=_,
                    instance=instance,
                    cluster_id=_,
                ):
                    alog.debug("Starting bootstrap for instance {iid}", iid=instance.id)

                    async def run_bootstrap() -> None:
                        from skyward.providers.bootstrap import (
                            run_bootstrap_via_ssh,
                            wait_for_ssh,
                        )

                        key_path = get_ssh_key_path()
                        ttl = state.spec.ttl or config.instance_timeout
                        bootstrap_script = state.spec.image.generate_bootstrap(
                            ttl=ttl,
                            shutdown_command=(
                                "eval $(cat /proc/1/environ "
                                "| tr '\\0' '\\n' "
                                "| grep -E 'CONTAINER_ID"
                                "|CONTAINER_API_KEY' "
                                "| sed 's/^/export /'); "
                                "curl -s -X DELETE "
                                "https://console.vast.ai"
                                "/api/v0/instances/"
                                "$CONTAINER_ID/ "
                                "-H \"Authorization: "
                                "Bearer $CONTAINER_API_KEY\""
                            ),
                        )

                        transport = await wait_for_ssh(
                            host=instance.ip,
                            user="root",
                            key_path=key_path,
                            timeout=state.spec.ssh_timeout,
                            poll_interval=state.spec.ssh_retry_interval,
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

                    ctx.pipe_to_self(run_bootstrap(), mapper=lambda x: x)  # type: ignore[reportArgumentType]
                    return Behaviors.same()

                case BootstrapDone(instance=info, success=True):
                    if state.spec.image and state.spec.image.includes:
                        ctx.pipe_to_self(
                            coro=_sync_user_code_vastai(info, state),
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
                        async with client:
                            async def _destroy(iid: str) -> None:
                                try:
                                    await client.destroy_instance(int(iid))
                                except Exception as e:
                                    alog.error("Failed to destroy {iid}: {err}", iid=iid, err=e)

                            await asyncio.gather(*(_destroy(iid) for iid in state.instances))

                            if state.overlay_name:
                                try:
                                    await client.delete_overlay(state.overlay_name)
                                except Exception as e:
                                    alog.error(
                                        "Failed to delete overlay {name}: {err}",
                                        name=state.overlay_name, err=e,
                                    )

                        await client.close()

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


# =============================================================================
# Internal Helpers
# =============================================================================


async def _wait_and_emit_running(
    client: VastAIClient,
    config: VastAI,
    state: VastAIClusterState,
    node_refs: dict[int, ActorRef],
    request_id: str,
    cluster_id: str,
    node_id: int,
    instance_id: int,
) -> None:
    """Wait for instance to be running, join overlay, detect IPs, emit InstanceRunning."""
    log = logger.bind(provider="vastai", instance_id=str(instance_id))
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
            log.error("Instance did not become ready")
            return

        if not info:
            log.error("Instance not found")
            return

        direct_port = get_direct_ssh_port(info)
        log.debug(
            "Connection info: public_ipaddr={pub_ip}, direct_port={dport}, "
            "ssh_host={shost}, ports={ports}",
            pub_ip=info.get("public_ipaddr"), dport=direct_port,
            shost=info.get("ssh_host"), ports=info.get("ports"),
        )
        if info.get("public_ipaddr") and direct_port:
            ssh_host = info["public_ipaddr"]
            ssh_port = direct_port
            log.info("Using direct IP {host}:{port}", host=ssh_host, port=ssh_port)
        else:
            ssh_host = info["ssh_host"]
            ssh_port = info.get("ssh_port", 22)
            log.warning("Falling back to SSH proxy {host}:{port}", host=ssh_host, port=ssh_port)

        private_ip = ""
        network_interface = ""

        if state.overlay_name:
            try:
                await client.join_overlay(state.overlay_name, instance_id)
                log.info("Joined overlay '{name}'", name=state.overlay_name)

                private_ip, network_interface = await _detect_overlay_ip(ssh_host, ssh_port)
            except VastAIError as e:
                log.error("Failed to join overlay '{name}': {err}", name=state.overlay_name, err=e)
                with suppress(Exception):
                    await client.destroy_instance(instance_id)
                return
            except RuntimeError as e:
                log.error("Failed to detect overlay IP: {err}", err=e)
                with suppress(Exception):
                    await client.destroy_instance(instance_id)
                return
        else:
            try:
                private_ip = await _detect_container_ip(ssh_host, ssh_port)
            except RuntimeError as e:
                log.warning("Could not detect container IP: {err}", err=e)
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

    key_path = get_ssh_key_path()
    running = InstanceRunning(
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
        ssh_user="root",
        ssh_key_path=key_path,
    )
    target = node_refs.get(node_id)
    if target:
        target.tell(running)


async def _sync_user_code_vastai(
    info: object,
    state: VastAIClusterState,
) -> None:
    from skyward.providers.bootstrap import sync_user_code

    key_path = get_ssh_key_path()
    await sync_user_code(
        host=info.ip,  # type: ignore[attr-defined]
        user="root",
        key_path=key_path,
        port=info.ssh_port,  # type: ignore[attr-defined]
        image=state.spec.image,
        use_sudo=False,
        ssh_timeout=state.spec.ssh_timeout,
        ssh_retry_interval=state.spec.ssh_retry_interval,
    )
