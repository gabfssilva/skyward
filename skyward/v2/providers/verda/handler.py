"""Verda Provider Handler - event-driven with Event Pipeline.

Uses intermediate events (InstanceLaunched, InstanceRunning) for
decoupled instance lifecycle management.
"""

from __future__ import annotations

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
from skyward.v2.providers.ssh_keys import ensure_ssh_key_on_provider, get_ssh_key_path
from skyward.v2.providers.wait import wait_for_ready

from .client import VerdaClient, VerdaError
from .config import Verda
from .state import VerdaClusterState
from .types import (
    InstanceResponse,
    InstanceTypeResponse,
    get_accelerator,
    get_price_on_demand,
    get_price_spot,
)

if TYPE_CHECKING:
    from skyward.v2.spec import PoolSpec


@component
class VerdaHandler:
    """Event-driven Verda provider using Event Pipeline.

    Flow:
        ClusterRequested → setup infra → ClusterProvisioned
        InstanceRequested → launch → InstanceLaunched
        InstanceLaunched → poll running → InstanceRunning
        BootstrapRequested → run bootstrap → (BootstrapPhase events)
        ShutdownRequested → cleanup → ClusterDestroyed

    The InstanceOrchestrator handles:
        InstanceRunning → InstanceProvisioned + BootstrapRequested
        BootstrapPhase(complete) → InstanceBootstrapped
    """

    bus: AsyncEventBus
    config: Verda
    client: VerdaClient

    _clusters: dict[str, VerdaClusterState] = field(default_factory=dict)

    # -------------------------------------------------------------------------
    # Cluster Lifecycle
    # -------------------------------------------------------------------------

    @on(ClusterRequested, match=lambda self, e: e.provider == "verda")
    async def handle_cluster_requested(self, _: Any, event: ClusterRequested) -> None:
        """Provision Verda infrastructure for a new cluster."""
        logger.info(f"Verda: Provisioning cluster for {event.spec.nodes} nodes")

        cluster_id = f"verda-{uuid.uuid4().hex[:8]}"

        state = VerdaClusterState(
            cluster_id=cluster_id,
            spec=event.spec,
            region=self.config.region,
        )
        self._clusters[cluster_id] = state

        async with self.client:
            # Ensure SSH key exists
            ssh_key_id = await ensure_ssh_key_on_provider(
                list_keys_fn=self.client.list_ssh_keys,
                create_key_fn=lambda name, key: self.client.create_ssh_key(name, key),
                provider_name="verda",
            )
            state.ssh_key_id = ssh_key_id

            # Resolve instance type and image
            instance_type, os_image = await self._resolve_instance_type(event.spec)
            state.instance_type = instance_type
            state.os_image = os_image

            # Create startup script
            user_data = self._generate_user_data(event.spec)
            script_name = f"skyward-bootstrap-{cluster_id}"
            script = await self.client.create_startup_script(script_name, user_data)
            state.startup_script_id = script["id"]

        self.bus.emit(
            ClusterProvisioned(
                request_id=event.request_id,
                cluster_id=cluster_id,
                provider="verda",
            )
        )

    @on(ShutdownRequested)
    async def handle_shutdown_requested(self, _: Any, event: ShutdownRequested) -> None:
        """Terminate all instances in a cluster."""
        cluster = self._clusters.pop(event.cluster_id, None)
        if not cluster:
            return

        logger.info(f"Verda: Shutting down cluster {event.cluster_id}")

        async with self.client:
            for instance_id in cluster.instance_ids:
                with suppress(Exception):
                    await self.client.delete_instance(instance_id)

            if cluster.startup_script_id:
                with suppress(Exception):
                    await self.client.delete_startup_script(cluster.startup_script_id)

        self.bus.emit(ClusterDestroyed(cluster_id=event.cluster_id))

    # -------------------------------------------------------------------------
    # Instance Lifecycle - Event Pipeline
    # -------------------------------------------------------------------------

    @on(InstanceRequested, match=lambda self, e: e.provider == "verda")
    async def handle_instance_requested(self, _: Any, event: InstanceRequested) -> None:
        """Launch Verda instance and emit InstanceLaunched."""
        cluster = self._clusters.get(event.cluster_id)
        if not cluster or not cluster.instance_type or not cluster.os_image:
            return

        logger.info(f"Verda: Launching instance for node {event.node_id}")

        use_spot = cluster.spec.allocation in ("spot", "spot-if-available")

        async with self.client:
            actual_region = await self._find_available_region(
                cluster.instance_type, use_spot, cluster.region
            )

            hostname = f"skyward-{cluster.cluster_id}-{event.node_id}"

            try:
                instance = await self.client.create_instance(
                    instance_type=cluster.instance_type,
                    image=cluster.os_image,
                    ssh_key_ids=[cluster.ssh_key_id] if cluster.ssh_key_id else [],
                    location=actual_region,
                    hostname=hostname,
                    description=f"Skyward managed - cluster {cluster.cluster_id}",
                    startup_script_id=cluster.startup_script_id,
                    is_spot=use_spot,
                )
            except VerdaError as e:
                logger.error(f"Verda: Failed to create instance: {e}")
                return

        # Track that we're waiting for this instance
        cluster.pending_nodes.add(event.node_id)

        # Emit intermediate event - instance created, waiting for running
        self.bus.emit(
            InstanceLaunched(
                request_id=event.request_id,
                cluster_id=event.cluster_id,
                node_id=event.node_id,
                provider="verda",
                instance_id=str(instance["id"]),
            )
        )

    @on(InstanceLaunched, match=lambda self, e: e.provider == "verda")
    async def handle_instance_launched(self, _: Any, event: InstanceLaunched) -> None:
        """Wait for instance to be running and emit InstanceRunning."""
        cluster = self._clusters.get(event.cluster_id)
        if not cluster:
            return

        use_spot = cluster.spec.allocation in ("spot", "spot-if-available")

        async with self.client:
            try:
                info = await wait_for_ready(
                    poll_fn=lambda: self.client.get_instance(event.instance_id),
                    ready_check=lambda i: i is not None and i["status"] == "running" and bool(i.get("ip")),
                    terminal_check=lambda i: i is not None and i["status"] in ("error", "discontinued", "deleted"),
                    timeout=300.0,
                    interval=5.0,
                    description=f"Verda instance {event.instance_id}",
                )
            except TimeoutError:
                logger.error(f"Verda: Instance {event.instance_id} did not become ready")
                return

        if not info:
            logger.error(f"Verda: Instance {event.instance_id} not found")
            return

        # Emit InstanceRunning - InstanceOrchestrator will handle the rest
        self.bus.emit(
            InstanceRunning(
                request_id=event.request_id,
                cluster_id=event.cluster_id,
                node_id=event.node_id,
                provider="verda",
                instance_id=event.instance_id,
                ip=info["ip"],
                private_ip=info.get("private_ip"),
                ssh_port=22,
                spot=use_spot,
            )
        )

    @on(BootstrapRequested, match=lambda self, e: e.instance.provider == "verda")
    async def handle_bootstrap_requested(self, _: Any, event: BootstrapRequested) -> None:
        """Execute bootstrap on instance. BootstrapPhase events are emitted automatically."""
        cluster = self._clusters.get(event.cluster_id)
        if not cluster:
            return

        from skyward.v2.providers.bootstrap import wait_bootstrap_with_streaming

        logger.debug(f"Verda: Starting bootstrap for instance {event.instance.id}")

        await wait_bootstrap_with_streaming(
            info=event.instance,
            bus=self.bus,
            user="root",  # Verda instances use root
            key_path=get_ssh_key_path(),
            timeout=600.0,
            poll_interval=5.0,
            log_prefix="Verda: ",
        )

        # Track instance in cluster state
        cluster.add_instance(event.instance)

        # Emit InstanceBootstrapped - Node will signal NodeReady
        self.bus.emit(InstanceBootstrapped(instance=event.instance))

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    async def _resolve_instance_type(self, spec: PoolSpec) -> tuple[str, str]:
        """Resolve instance type and OS image from spec."""
        use_spot = spec.allocation in ("spot", "spot-if-available")

        instance_types = await self.client.list_instance_types()
        availability = await self.client.get_availability(is_spot=use_spot)

        available_types: set[str] = set()
        for region_types in availability.values():
            available_types.update(region_types)

        candidates: list[InstanceTypeResponse] = []
        for itype in instance_types:
            if itype["instance_type"] not in available_types:
                continue

            if spec.accelerator_name:
                accel = get_accelerator(itype)
                if not accel:
                    continue
                # Compare base model (e.g., "A100" matches "A100-80GB" or "A100")
                accel_upper = accel.upper()
                requested_upper = spec.accelerator_name.upper()
                if accel_upper not in requested_upper and requested_upper not in accel_upper:
                    continue

            candidates.append(itype)

        if not candidates:
            raise RuntimeError(f"No instance types match accelerator={spec.accelerator_name}")

        def sort_key(it: InstanceTypeResponse) -> float:
            price = get_price_spot(it) if use_spot else get_price_on_demand(it)
            return price if price is not None else float("inf")

        candidates.sort(key=sort_key)

        selected = candidates[0]
        supported_os = selected.get("supported_os", [])
        logger.debug(f"Verda: Instance {selected['instance_type']} supported_os: {supported_os}")

        os_image = self._select_os_image(spec, supported_os)

        logger.debug(f"Verda: Selected {selected['instance_type']} with image {os_image}")
        return selected["instance_type"], os_image

    def _select_os_image(self, spec: PoolSpec, supported_os: list[str]) -> str:
        """Select best OS image from supported list."""
        default_cuda_image = "ubuntu-22.04-cuda-12.1"

        if not spec.accelerator_name:
            return supported_os[0] if supported_os else "ubuntu-22.04"

        # Prefer ubuntu-based CUDA images
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
            """Parse image name to (cuda_major, cuda_minor, ubuntu_major, ubuntu_minor)."""
            import re
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
        self, instance_type: str, is_spot: bool, preferred_region: str
    ) -> str:
        """Find a region where the instance type is available."""
        availability = await self.client.get_availability(is_spot)

        if preferred_region in availability and instance_type in availability[preferred_region]:
            return preferred_region

        for region, types in availability.items():
            if instance_type in types:
                logger.info(f"Verda: Auto-selected region {region}")
                return region

        raise RuntimeError(f"No region has instance type '{instance_type}' available")

    def _generate_user_data(self, spec: PoolSpec) -> str:
        """Generate bootstrap user data script."""
        return spec.image.generate_bootstrap(ttl=spec.ttl, use_systemd=True)


__all__ = ["VerdaHandler"]
