"""InstanceOrchestrator - generic instance lifecycle orchestration.

This component handles the Event Pipeline flow, coordinating
instance lifecycle for ANY provider through events:

1. InstanceRunning → builds InstanceInfo → emits InstanceProvisioned
2. InstanceProvisioned → emits BootstrapRequested
3. BootstrapPhase(complete) → emits InstanceBootstrapped

Providers only need to emit InstanceLaunched and InstanceRunning,
then handle BootstrapRequested. The orchestrator handles the rest.
"""

from __future__ import annotations

from typing import Any

from .app import component, on
from .bus import AsyncEventBus
from .events import (
    BootstrapRequested,
    InstanceInfo,
    InstanceProvisioned,
    InstanceRunning,
)


@component
class InstanceOrchestrator:
    """Orchestrates instance lifecycle for all providers.

    This generic component transforms intermediate events into
    standard lifecycle events, eliminating duplicate code in
    provider handlers.

    Flow:
        InstanceRunning → InstanceProvisioned + BootstrapRequested

    Note: Each provider's handle_bootstrap_requested is responsible for
    emitting InstanceBootstrapped after bootstrap completes (this allows
    providers to do post-bootstrap work like installing local wheel).
    """

    bus: AsyncEventBus

    @on(InstanceRunning)
    async def on_instance_running(
        self,
        _sender: Any,
        event: InstanceRunning,
    ) -> None:
        """Handle instance running - build InstanceInfo and emit events.

        When an instance is running:
        1. Build immutable InstanceInfo
        2. Emit InstanceProvisioned (Node updates its state)
        3. Emit BootstrapRequested (provider runs bootstrap)
        """
        # Build InstanceInfo from event data
        info = InstanceInfo(
            id=event.instance_id,
            node=event.node_id,
            provider=event.provider,
            ip=event.ip,
            private_ip=event.private_ip or "",
            network_interface=event.network_interface,
            spot=event.spot,
            ssh_port=event.ssh_port,
            # Pricing info from provider
            hourly_rate=event.hourly_rate,
            on_demand_rate=event.on_demand_rate,
            billing_increment=event.billing_increment,
            # Instance details from provider
            instance_type=event.instance_type,
            gpu_count=event.gpu_count,
            gpu_model=event.gpu_model,
            # Hardware specs from provider
            vcpus=event.vcpus,
            memory_gb=event.memory_gb,
            gpu_vram_gb=event.gpu_vram_gb,
            # Location info from provider
            region=event.region,
        )

        # Emit InstanceProvisioned - Node receives this
        # Use emit_await to ensure Node updates state before bootstrap starts
        await self.bus.emit_await(
            InstanceProvisioned(
                request_id=event.request_id,
                instance=info,
            )
        )

        # Emit BootstrapRequested - provider handles this
        self.bus.emit(
            BootstrapRequested(
                request_id=event.request_id,
                instance=info,
                cluster_id=event.cluster_id,
            )
        )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "InstanceOrchestrator",
]
