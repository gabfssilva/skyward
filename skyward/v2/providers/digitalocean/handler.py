"""DigitalOcean Provider Handler - event-driven with Event Pipeline.

Uses intermediate events (InstanceLaunched, InstanceRunning) for
decoupled instance lifecycle management.
"""

from __future__ import annotations

import os
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
from skyward.v2.providers.ssh_keys import (
    compute_fingerprint,
    get_local_ssh_key,
    get_ssh_key_path,
)
from skyward.v2.providers.wait import wait_for_ready

from .client import DigitalOceanClient, DigitalOceanError
from .config import DigitalOcean
from .state import DOClusterState
from .types import (
    DropletResponse,
    SizeResponse,
    get_gpu_image,
    get_private_ip,
    get_public_ip,
    normalize_gpu_model,
)

if TYPE_CHECKING:
    from skyward.v2.spec import PoolSpec


async def _ensure_ssh_key(client: DigitalOceanClient, config: DigitalOcean) -> str:
    """Get or create SSH key on DigitalOcean. Returns fingerprint.

    DigitalOcean uses fingerprint as the key identifier.
    """
    # Check config/env first
    if config.ssh_key_fingerprint:
        return config.ssh_key_fingerprint

    env_fingerprint = os.environ.get("DIGITALOCEAN_SSH_KEY_FINGERPRINT")
    if env_fingerprint:
        return env_fingerprint

    # Get local SSH key
    public_path, public_key = get_local_ssh_key()
    local_fingerprint = compute_fingerprint(public_key)

    # Check if already registered
    for key in await client.list_ssh_keys():
        if key["fingerprint"] == local_fingerprint:
            return key["fingerprint"]

    # Register new key
    key_name = f"skyward-{os.environ.get('USER', 'user')}-{public_path.stem}"
    logger.info(f"DigitalOcean: Creating SSH key: {key_name}")

    try:
        new_key = await client.create_ssh_key(key_name, public_key)
        return new_key["fingerprint"]
    except DigitalOceanError as e:
        if "already been taken" in str(e).lower():
            for key in await client.list_ssh_keys():
                if key["fingerprint"] == local_fingerprint:
                    return key["fingerprint"]
        raise


@component
class DOHandler:
    """Event-driven DigitalOcean provider using Event Pipeline.

    Flow:
        ClusterRequested → setup infra → ClusterProvisioned
        InstanceRequested → launch → InstanceLaunched
        InstanceLaunched → poll active → InstanceRunning
        BootstrapRequested → run bootstrap → (BootstrapPhase events)
        ShutdownRequested → cleanup → ClusterDestroyed

    The InstanceOrchestrator handles:
        InstanceRunning → InstanceProvisioned + BootstrapRequested
        BootstrapPhase(complete) → InstanceBootstrapped
    """

    bus: AsyncEventBus
    config: DigitalOcean
    client: DigitalOceanClient

    _clusters: dict[str, DOClusterState] = field(default_factory=dict)

    # -------------------------------------------------------------------------
    # Cluster Lifecycle
    # -------------------------------------------------------------------------

    @on(ClusterRequested, match=lambda self, e: e.provider == "digitalocean")
    async def handle_cluster_requested(self, _: Any, event: ClusterRequested) -> None:
        """Provision DigitalOcean infrastructure for a new cluster."""
        logger.info(f"DigitalOcean: Provisioning cluster for {event.spec.nodes} nodes")

        cluster_id = f"do-{uuid.uuid4().hex[:8]}"

        state = DOClusterState(
            cluster_id=cluster_id,
            spec=event.spec,
            region=self.config.region,
        )
        self._clusters[cluster_id] = state

        async with self.client:
            fingerprint = await _ensure_ssh_key(self.client, self.config)
            state.ssh_key_fingerprint = fingerprint

            size_slug, os_image, username = await self._resolve_instance_config(event.spec)
            state.size_slug = size_slug
            state.os_image = os_image
            state.username = username

        self.bus.emit(
            ClusterProvisioned(
                request_id=event.request_id,
                cluster_id=cluster_id,
                provider="digitalocean",
            )
        )

    @on(ShutdownRequested)
    async def handle_shutdown_requested(self, _: Any, event: ShutdownRequested) -> None:
        """Terminate all droplets in a cluster."""
        cluster = self._clusters.pop(event.cluster_id, None)
        if not cluster:
            return

        logger.info(f"DigitalOcean: Shutting down cluster {event.cluster_id}")

        async with self.client:
            for droplet_id in cluster.instance_ids:
                with suppress(Exception):
                    await self.client.delete_droplet(droplet_id)

        self.bus.emit(ClusterDestroyed(cluster_id=event.cluster_id))

    # -------------------------------------------------------------------------
    # Instance Lifecycle - Event Pipeline
    # -------------------------------------------------------------------------

    @on(InstanceRequested, match=lambda self, e: e.provider == "digitalocean")
    async def handle_instance_requested(self, _: Any, event: InstanceRequested) -> None:
        """Launch DigitalOcean droplet and emit InstanceLaunched."""
        cluster = self._clusters.get(event.cluster_id)
        if not cluster or not cluster.size_slug or not cluster.os_image:
            return

        logger.info(f"DigitalOcean: Launching droplet for node {event.node_id}")

        async with self.client:
            droplet_name = f"skyward-{cluster.cluster_id}-{event.node_id}"
            tags = ["skyward", f"skyward-cluster-{cluster.cluster_id}"]

            user_data = self._generate_user_data(cluster.spec)

            try:
                droplet = await self.client.create_droplet(
                    name=droplet_name,
                    region=cluster.region,
                    size=cluster.size_slug,
                    image=cluster.os_image,
                    ssh_keys=[cluster.ssh_key_fingerprint],
                    user_data=user_data,
                    tags=tags,
                )
            except DigitalOceanError as e:
                logger.error(f"DigitalOcean: Failed to create droplet: {e}")
                return

        # Track that we're waiting for this droplet
        cluster.pending_nodes.add(event.node_id)

        # Emit intermediate event - droplet created, waiting for active
        self.bus.emit(
            InstanceLaunched(
                request_id=event.request_id,
                cluster_id=event.cluster_id,
                node_id=event.node_id,
                provider="digitalocean",
                instance_id=str(droplet["id"]),
            )
        )

    @on(InstanceLaunched, match=lambda self, e: e.provider == "digitalocean")
    async def handle_instance_launched(self, _: Any, event: InstanceLaunched) -> None:
        """Wait for droplet to be active and emit InstanceRunning."""
        cluster = self._clusters.get(event.cluster_id)
        if not cluster:
            return

        async with self.client:
            try:
                info = await wait_for_ready(
                    poll_fn=lambda: self.client.get_droplet(int(event.instance_id)),
                    ready_check=lambda d: d is not None and d.get("status") == "active" and bool(get_public_ip(d)),
                    terminal_check=lambda d: d is not None and d.get("status") in ("archive", "off"),
                    timeout=300.0,
                    interval=5.0,
                    description=f"DigitalOcean droplet {event.instance_id}",
                )
            except TimeoutError:
                logger.error(f"DigitalOcean: Droplet {event.instance_id} did not become active")
                return

        if not info:
            logger.error(f"DigitalOcean: Droplet {event.instance_id} not found")
            return

        public_ip = get_public_ip(info) or ""
        private_ip = get_private_ip(info) or public_ip

        # Emit InstanceRunning - InstanceOrchestrator will handle the rest
        self.bus.emit(
            InstanceRunning(
                request_id=event.request_id,
                cluster_id=event.cluster_id,
                node_id=event.node_id,
                provider="digitalocean",
                instance_id=event.instance_id,
                ip=public_ip,
                private_ip=private_ip,
                ssh_port=22,
                spot=False,  # DigitalOcean doesn't have spot instances
            )
        )

    @on(BootstrapRequested, match=lambda self, e: e.instance.provider == "digitalocean")
    async def handle_bootstrap_requested(self, _: Any, event: BootstrapRequested) -> None:
        """Execute bootstrap on droplet. BootstrapPhase events are emitted automatically."""
        cluster = self._clusters.get(event.cluster_id)
        if not cluster:
            return

        from skyward.v2.providers.bootstrap import wait_bootstrap_with_streaming

        # Get username from cluster state (determined during instance config resolution)
        username = cluster.username or "root"

        logger.debug(f"DigitalOcean: Starting bootstrap for droplet {event.instance.id}")

        await wait_bootstrap_with_streaming(
            info=event.instance,
            bus=self.bus,
            user=username,
            key_path=get_ssh_key_path(),
            timeout=600.0,
            poll_interval=5.0,
            log_prefix="DigitalOcean: ",
        )

        # Track instance in cluster state
        cluster.add_instance(event.instance)

        # Emit InstanceBootstrapped - Node will signal NodeReady
        self.bus.emit(InstanceBootstrapped(instance=event.instance))

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    async def _resolve_instance_config(
        self, spec: PoolSpec
    ) -> tuple[str, str, str]:
        """Resolve size slug, image, and username from spec."""
        sizes = await self.client.list_sizes()
        available = [s for s in sizes if s.get("available", True)]

        candidates: list[SizeResponse] = []
        accelerator_count = 0

        for size in available:
            gpu_info = size.get("gpu_info")

            if spec.accelerator_name:
                if not gpu_info:
                    continue
                gpu_model = normalize_gpu_model(gpu_info.get("model"))
                if not gpu_model or spec.accelerator_name.upper() not in gpu_model:
                    continue
                accelerator_count = gpu_info.get("count", 1)

            candidates.append(size)

        if not candidates:
            raise RuntimeError(f"No sizes match accelerator={spec.accelerator_name}")

        # Sort by price
        candidates.sort(key=lambda s: s.get("price_hourly", float("inf")))
        selected = candidates[0]

        # Determine image and username
        gpu_info = selected.get("gpu_info")
        if gpu_info:
            os_image = get_gpu_image(spec.accelerator_name, accelerator_count)
            username = "ubuntu"
        else:
            os_image = "ubuntu-24-04-x64"
            username = "root"

        logger.debug(f"DigitalOcean: Selected {selected['slug']} with image {os_image}")
        return selected["slug"], os_image, username

    def _generate_user_data(self, spec: PoolSpec) -> str:
        """Generate bootstrap user data script."""
        return spec.image.generate_bootstrap(ttl=spec.ttl, use_systemd=True)


__all__ = ["DOHandler"]
