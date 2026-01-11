"""Vast.ai Provider Handler - event-driven with Event Pipeline.

Uses intermediate events (InstanceLaunched, InstanceRunning) for
decoupled instance lifecycle management.

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
from dataclasses import field
from typing import TYPE_CHECKING, Any

from loguru import logger

from skyward.v2.app import component, on
from skyward.v2.bus import AsyncEventBus
from skyward.v2.events import (
    BootstrapRequested,
    ClusterDestroyed,
    ClusterProvisioned,
    ClusterRequested,
    InstanceBootstrapped,
    InstanceLaunched,
    InstanceRunning,
    InstanceRequested,
    ShutdownRequested,
)
from skyward.v2.monitors import SSHCredentialsRegistry
from skyward.v2.providers.ssh_keys import get_ssh_key_path
from skyward.v2.providers.wait import wait_for_ready

from .client import VastAIClient, VastAIError, select_all_valid_clusters
from .config import VastAI
from .state import InstancePricing, VastAIClusterState
from .types import InstanceResponse, OfferResponse, get_direct_ssh_port

if TYPE_CHECKING:
    from skyward.v2.spec import PoolSpec


@component
class VastAIHandler:
    """Event-driven Vast.ai provider using Event Pipeline.

    Flow:
        ClusterRequested → setup infra/overlay → ClusterProvisioned
        InstanceRequested → create instance → InstanceLaunched
        InstanceLaunched → wait running, join overlay → InstanceRunning
        BootstrapRequested → run bootstrap via SSH → (BootstrapPhase events)
        ShutdownRequested → cleanup → ClusterDestroyed

    The InstanceOrchestrator handles:
        InstanceRunning → InstanceProvisioned + BootstrapRequested
        BootstrapPhase(complete) → InstanceBootstrapped
    """

    bus: AsyncEventBus
    config: VastAI
    client: VastAIClient
    ssh_credentials: SSHCredentialsRegistry

    _clusters: dict[str, VastAIClusterState] = field(default_factory=dict)
    _reserved_offers: set[int] = field(default_factory=set)  # offer_ids in use

    # -------------------------------------------------------------------------
    # Cluster Lifecycle
    # -------------------------------------------------------------------------

    @on(ClusterRequested, match=lambda self, e: e.provider == "vastai")
    async def handle_cluster_requested(self, _: Any, event: ClusterRequested) -> None:
        """Provision Vast.ai infrastructure for a new cluster."""
        logger.info(f"VastAI: Provisioning cluster for {event.spec.nodes} nodes")

        cluster_id = f"vastai-{uuid.uuid4().hex[:8]}"

        state = VastAIClusterState(
            cluster_id=cluster_id,
            spec=event.spec,
            geolocation=self.config.geolocation,
        )
        self._clusters[cluster_id] = state

        async with self.client:
            ssh_key_id, ssh_public_key = await self.client.ensure_ssh_key()
            state.ssh_key_id = ssh_key_id
            state.ssh_public_key = ssh_public_key

            if event.spec.nodes > 1 and self.config.use_overlay:
                await self._setup_overlay_network(state, event.spec)

        # Register SSH credentials for EventStreamer
        ssh_key_path = get_ssh_key_path()
        self.ssh_credentials.register(cluster_id, "root", ssh_key_path)

        self.bus.emit(
            ClusterProvisioned(
                request_id=event.request_id,
                cluster_id=cluster_id,
                provider="vastai",
            )
        )

    @on(ShutdownRequested)
    async def handle_shutdown_requested(self, _: Any, event: ShutdownRequested) -> None:
        """Terminate all instances in a cluster."""
        cluster = self._clusters.pop(event.cluster_id, None)
        if not cluster:
            return

        logger.info(f"VastAI: Shutting down cluster {event.cluster_id}")

        async with self.client:
            for instance_id in cluster.instance_ids:
                with suppress(Exception):
                    await self.client.destroy_instance(int(instance_id))

            if cluster.overlay_name:
                with suppress(Exception):
                    await self.client.delete_overlay(cluster.overlay_name)

        # Close HTTP client when no more clusters
        if not self._clusters:
            await self.client.close()

        self.bus.emit(ClusterDestroyed(cluster_id=event.cluster_id))

    # -------------------------------------------------------------------------
    # Instance Lifecycle - Event Pipeline
    # -------------------------------------------------------------------------

    @on(InstanceRequested, match=lambda self, e: e.provider == "vastai")
    async def handle_instance_requested(self, _: Any, event: InstanceRequested) -> None:
        """Create VastAI instance and emit InstanceLaunched."""
        cluster = self._clusters.get(event.cluster_id)
        if not cluster:
            return

        logger.info(f"VastAI: Launching instance for node {event.node_id}")

        async with self.client:
            offers = await self._search_offers(cluster)

            if not offers:
                logger.error(f"VastAI: No offers found for node {event.node_id}")
                return

            use_interruptible = cluster.spec.allocation in ("spot", "spot-if-available")
            docker_image = cluster.docker_image or self.config.docker_image or VastAI.ubuntu()
            label = f"skyward-{cluster.cluster_id}-{event.node_id}"

            # Minimal onstart - full bootstrap via SSH
            minimal_onstart = "#!/bin/bash\nset -e\nmkdir -p /opt/skyward\ntail -f /dev/null\n"

            # Try each offer until one succeeds
            instance_id: int | None = None
            last_error: str | None = None

            for idx, offer in enumerate(offers):
                offer_id = offer["id"]

                if offer_id in self._reserved_offers:
                    logger.debug(f"VastAI: Offer {offer_id} already reserved, skipping...")
                    continue

                self._reserved_offers.add(offer_id)

                price = offer["min_bid"] * self.config.bid_multiplier if use_interruptible else None
                price_display = price if price else offer.get("dph_total", 0)

                logger.info(
                    f"VastAI: Trying offer {idx + 1}/{len(offers)}: "
                    f"machine_id={offer.get('machine_id')}, price=${price_display:.3f}/hr"
                )

                try:
                    instance_id = await self.client.create_instance(
                        offer_id=offer_id,
                        image=docker_image,
                        disk=self.config.disk_gb,
                        label=label,
                        onstart_cmd=minimal_onstart,
                        price=price,
                    )
                    # Store pricing info for use in InstanceRunning
                    on_demand_rate = offer.get("dph_total", 0.0)
                    hourly_rate = price if price else on_demand_rate
                    cluster.instance_pricing[str(instance_id)] = InstancePricing(
                        hourly_rate=hourly_rate,
                        on_demand_rate=on_demand_rate,
                        gpu_name=offer.get("gpu_name", ""),
                        gpu_count=offer.get("num_gpus", 0),
                    )
                    break
                except VastAIError as e:
                    self._reserved_offers.discard(offer_id)
                    last_error = str(e)
                    logger.warning(f"VastAI: Offer {idx + 1}/{len(offers)} failed: {e}")
                    continue

            if instance_id is None:
                logger.error(
                    f"VastAI: All {len(offers)} offers failed for node {event.node_id}. "
                    f"Last error: {last_error}"
                )
                return

            await self.client.attach_ssh(instance_id, cluster.ssh_public_key)

        # Track pending
        cluster.pending_nodes.add(event.node_id)

        # Emit intermediate event
        self.bus.emit(
            InstanceLaunched(
                request_id=event.request_id,
                cluster_id=event.cluster_id,
                node_id=event.node_id,
                provider="vastai",
                instance_id=str(instance_id),
            )
        )

    @on(InstanceLaunched, match=lambda self, e: e.provider == "vastai")
    async def handle_instance_launched(self, _: Any, event: InstanceLaunched) -> None:
        """Wait for running, join overlay, detect IPs, emit InstanceRunning."""
        cluster = self._clusters.get(event.cluster_id)
        if not cluster:
            return

        use_interruptible = cluster.spec.allocation in ("spot", "spot-if-available")
        instance_id = int(event.instance_id)

        async with self.client:
            try:
                info = await wait_for_ready(
                    poll_fn=lambda: self.client.get_instance(instance_id),
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
                    description=f"VastAI instance {event.instance_id}",
                )
            except TimeoutError:
                logger.error(f"VastAI: Instance {event.instance_id} did not become ready")
                return

            if not info:
                logger.error(f"VastAI: Instance {event.instance_id} not found")
                return

            # Get SSH connection info
            direct_port = get_direct_ssh_port(info)
            if info["public_ipaddr"] and direct_port:
                ssh_host = info["public_ipaddr"]
                ssh_port = direct_port
            else:
                ssh_host = info["ssh_host"]
                ssh_port = info.get("ssh_port", 22)

            # Join overlay and detect IPs
            private_ip = ""
            network_interface = ""

            if cluster.overlay_name:
                try:
                    await self.client.join_overlay(cluster.overlay_name, instance_id)
                    logger.info(f"VastAI: Instance {instance_id} joined overlay '{cluster.overlay_name}'")

                    private_ip, network_interface = await self._detect_overlay_ip(ssh_host, ssh_port)
                except VastAIError as e:
                    logger.error(f"VastAI: Failed to join overlay '{cluster.overlay_name}': {e}")
                    with suppress(Exception):
                        await self.client.destroy_instance(instance_id)
                    return
                except RuntimeError as e:
                    logger.error(f"VastAI: Failed to detect overlay IP: {e}")
                    with suppress(Exception):
                        await self.client.destroy_instance(instance_id)
                    return
            else:
                # Single node: detect container IP for local coordination
                try:
                    private_ip = await self._detect_container_ip(ssh_host, ssh_port)
                except RuntimeError as e:
                    logger.warning(f"VastAI: Could not detect container IP: {e}")
                    private_ip = ssh_host

        # Retrieve pricing info stored during instance creation
        pricing = cluster.instance_pricing.get(event.instance_id)
        if pricing:
            hourly_rate = pricing.hourly_rate
            on_demand_rate = pricing.on_demand_rate
            gpu_name = pricing.gpu_name
            gpu_count = pricing.gpu_count
        else:
            # Fallback: use info from instance API (no spot rate available)
            hourly_rate = info.get("dph_total", 0.0)
            on_demand_rate = hourly_rate
            gpu_name = info.get("gpu_name", "")
            gpu_count = info.get("num_gpus", 0)

        # Get hardware specs from VastAI instance info
        vcpus = int(info.get("cpu_cores_effective", 0))
        memory_gb = info.get("cpu_ram", 0) / 1024  # MB to GB
        # gpu_ram is total VRAM in MB, divide by gpu_count for per-GPU
        total_vram_mb = info.get("gpu_ram", 0)
        gpu_vram_gb = int(total_vram_mb / 1024 / gpu_count) if gpu_count else 0

        # Emit InstanceRunning - InstanceOrchestrator handles the rest
        self.bus.emit(
            InstanceRunning(
                request_id=event.request_id,
                cluster_id=event.cluster_id,
                node_id=event.node_id,
                provider="vastai",
                instance_id=event.instance_id,
                ip=ssh_host,
                private_ip=private_ip,
                ssh_port=ssh_port,
                spot=use_interruptible,
                network_interface=network_interface,
                # Pricing info from offer
                hourly_rate=hourly_rate,
                on_demand_rate=on_demand_rate,
                billing_increment=1,  # VastAI bills per-second
                instance_type=gpu_name,
                gpu_count=gpu_count,
                gpu_model=gpu_name,
                # Hardware specs from instance API
                vcpus=vcpus,
                memory_gb=memory_gb,
                gpu_vram_gb=gpu_vram_gb,
            )
        )

    @on(BootstrapRequested, match=lambda self, e: e.instance.provider == "vastai")
    async def handle_bootstrap_requested(self, _: Any, event: BootstrapRequested) -> None:
        """Execute bootstrap via SSH. BootstrapPhase events are emitted automatically."""
        cluster = self._clusters.get(event.cluster_id)
        if not cluster:
            return

        logger.debug(f"VastAI: Starting bootstrap for instance {event.instance.id}")

        # Run bootstrap via SSH
        await self._run_bootstrap_via_ssh(
            event.instance,
            cluster.spec,
            event.instance.ip,
            event.instance.ssh_port,
        )

        # Install local skyward wheel if skyward_source == 'local'
        if cluster.spec.image.skyward_source == "local":
            await self._install_local_skyward(
                event.instance,
                cluster.spec,
                event.instance.ip,
                event.instance.ssh_port,
            )

        # Track instance in cluster state
        cluster.add_instance(event.instance)

        # Emit InstanceBootstrapped - Node will signal NodeReady
        self.bus.emit(InstanceBootstrapped(instance=event.instance))

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    async def _search_offers(self, cluster: VastAIClusterState) -> list[OfferResponse]:
        """Search for GPU offers matching cluster spec."""
        spec = cluster.spec
        use_interruptible = spec.allocation in ("spot", "spot-if-available")
        gpu_name = spec.accelerator_name.replace(" ", "_").replace("-", "_") if spec.accelerator_name else None

        offers = await self.client.search_offers(
            gpu_name=gpu_name,
            min_reliability=self.config.min_reliability,
            geolocation=self.config.geolocation,
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

        # Filter to overlay cluster for multi-node
        if cluster.overlay_cluster_id is not None:
            before_cluster_filter = len(offers)
            offers = [o for o in offers if o.get("cluster_id") == cluster.overlay_cluster_id]
            logger.debug(
                f"VastAI: Cluster filter: {len(offers)}/{before_cluster_filter} offers "
                f"in overlay cluster {cluster.overlay_cluster_id}"
            )
            if not offers:
                logger.error(
                    f"VastAI: No offers in overlay cluster {cluster.overlay_cluster_id}. "
                    f"Cluster may have become unavailable."
                )

        # Sort by price
        price_key = "min_bid" if use_interruptible else "dph_total"
        offers.sort(key=lambda o: o.get(price_key, float("inf")))

        # Filter by budget
        if spec.max_hourly_cost:
            max_per_instance = spec.max_hourly_cost / spec.nodes

            def offer_price(o: OfferResponse) -> float:
                if use_interruptible:
                    return o.get("min_bid", float("inf")) * self.config.bid_multiplier
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
        self, cluster: VastAIClusterState, spec: PoolSpec
    ) -> None:
        """Set up overlay network for multi-node cluster."""
        use_interruptible = spec.allocation in ("spot", "spot-if-available")
        gpu_name = spec.accelerator_name.replace(" ", "_").replace("-", "_") if spec.accelerator_name else None

        offers = await self.client.search_offers(
            gpu_name=gpu_name,
            min_reliability=self.config.min_reliability,
            geolocation=self.config.geolocation,
            use_interruptible=use_interruptible,
            with_cluster_id=True,
        )

        valid_clusters = select_all_valid_clusters(offers, spec.nodes, use_interruptible)
        if not valid_clusters:
            logger.warning(f"VastAI: No clusters found with {spec.nodes} nodes")
            return

        for idx, (physical_cluster_id, _) in enumerate(valid_clusters):
            suffix = "".join(random.choices(string.ascii_lowercase, k=8))
            overlay_name = f"skyward-{suffix}"

            logger.info(f"VastAI: Trying cluster {physical_cluster_id} ({idx + 1}/{len(valid_clusters)})")

            try:
                await self.client.create_overlay(physical_cluster_id, overlay_name)
                cluster.overlay_name = overlay_name
                cluster.overlay_cluster_id = physical_cluster_id
                logger.info(f"VastAI: Overlay '{overlay_name}' created")
                return
            except VastAIError as e:
                logger.warning(f"VastAI: Overlay failed on cluster {physical_cluster_id}: {e}")

        logger.warning("VastAI: Failed to create overlay on any cluster")

    async def _detect_container_ip(
        self,
        ssh_host: str,
        ssh_port: int,
        timeout: float = 60.0,
    ) -> str:
        """Detect container's internal IP via SSH."""
        from skyward.v2.providers.bootstrap import wait_for_ssh

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
        self,
        ssh_host: str,
        ssh_port: int,
        timeout: float = 120.0,
        max_retries: int = 30,
    ) -> tuple[str, str]:
        """Detect overlay network IP (10.x.x.x) and interface via SSH."""
        from skyward.v2.providers.bootstrap import wait_for_ssh

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

    async def _run_bootstrap_via_ssh(
        self,
        instance_info: Any,
        spec: PoolSpec,
        ssh_host: str,
        ssh_port: int,
    ) -> None:
        """Run bootstrap script via SSH and stream events."""
        from skyward.v2.providers.bootstrap import (
            stream_bootstrap_events,
            wait_for_ssh,
        )

        key_path = get_ssh_key_path()

        logger.info(f"VastAI: Waiting for SSH on {ssh_host}:{ssh_port}...")
        transport = await wait_for_ssh(
            host=ssh_host,
            user="root",
            key_path=key_path,
            timeout=120.0,
            port=ssh_port,
            log_prefix="VastAI: ",
        )

        bootstrap_script = spec.image.generate_bootstrap(ttl=spec.ttl, use_systemd=False)
        logger.debug(f"VastAI: Bootstrap script ({len(bootstrap_script)} chars)")

        try:
            await transport.write_file("/opt/skyward/bootstrap.sh", bootstrap_script)
            await transport.run("chmod +x /opt/skyward/bootstrap.sh")

            logger.info(f"VastAI: Running bootstrap on {instance_info.id}...")
            await transport.run(
                "nohup /opt/skyward/bootstrap.sh > /opt/skyward/bootstrap.log 2>&1 &"
            )

            await asyncio.sleep(2)

            logger.info(f"VastAI: Streaming bootstrap events for {instance_info.id}...")
            await stream_bootstrap_events(
                transport=transport,
                info=instance_info,
                bus=self.bus,
                log_prefix="VastAI: ",
            )
        finally:
            await transport.close()

    async def _install_local_skyward(
        self,
        instance_info: Any,
        spec: PoolSpec,
        ssh_host: str,
        ssh_port: int,
    ) -> None:
        """Install local skyward wheel and start RPyC server."""
        from skyward.v2.providers.bootstrap import install_local_skyward, wait_for_ssh

        key_path = get_ssh_key_path()
        env = spec.image.env if spec.image else None

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
                env=env,
                use_systemd=False,
                rpyc_timeout=60.0,
                log_prefix="VastAI: ",
            )
        finally:
            await transport.close()


__all__ = ["VastAIHandler"]
