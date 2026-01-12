"""Panel component - Rich terminal dashboard using v2 event system.

Subscribes to v2 events via @on decorators and updates PanelState
which is rendered by Rich Live at 4fps.
"""

from __future__ import annotations

import time
from dataclasses import field

from skyward.app import component, on
from skyward.bus import AsyncEventBus
from skyward.events import (
    BootstrapCommand,
    BootstrapConsole,
    BootstrapPhase,
    ClusterProvisioned,
    ClusterReady,
    ClusterRequested,
    Error,
    InstanceBootstrapped,
    InstanceDestroyed,
    InstancePreempted,
    InstanceProvisioned,
    Log,
    Metric,
    ShutdownRequested,
)
from skyward.spec import PoolSpec

from .renderer import PanelRenderer
from .state import InfraState, InstanceState, MetricsState, PanelState


@component
class PanelComponent:
    """Rich terminal dashboard for cluster monitoring.

    Subscribes to v2 events and updates PanelState. The PanelRenderer
    uses Rich Live to refresh the display at 4fps.

    Features:
    - Real-time metrics with EMA smoothing and sparklines
    - Per-instance logs with Keras progress bar support
    - Cost tracking with billing increment support
    - Preemption detection and warning
    """

    bus: AsyncEventBus
    spec: PoolSpec

    _state: PanelState = field(default_factory=PanelState)
    _renderer: PanelRenderer = field(default_factory=PanelRenderer)
    _active: bool = False

    # =========================================================================
    # Cluster Lifecycle Events
    # =========================================================================

    @on(ClusterRequested)
    async def _on_cluster_requested(self, _sender: object, event: ClusterRequested) -> None:
        """Initialize panel when cluster is requested."""
        self._state.start_time = time.monotonic()
        self._state.total_nodes = event.spec.nodes
        self._state.phase = "Provisioning"
        self._state.is_done = False
        self._state.has_error = False
        self._state.instances.clear()

        # Initialize placeholders
        for i in range(event.spec.nodes):
            self._state.instances[f"node-{i}"] = InstanceState(f"node-{i}")

        self._renderer.start(self._state)
        self._active = True

    @on(ClusterProvisioned)
    async def _on_cluster_provisioned(self, _sender: object, event: ClusterProvisioned) -> None:
        """Mark provisioning complete."""
        self._state.phase = "Bootstrapping"
        elapsed = time.monotonic() - self._state.start_time if self._state.start_time else 0.0
        self._state.phase_times["provision"] = elapsed

    @on(ClusterReady)
    async def _on_cluster_ready(self, _sender: object, event: ClusterReady) -> None:
        """Mark cluster ready for execution."""
        self._state.phase = "Executing"
        self._state.ready = len(event.nodes)
        elapsed = time.monotonic() - self._state.start_time if self._state.start_time else 0.0
        self._state.phase_times["bootstrap"] = elapsed

    @on(ShutdownRequested)
    async def _on_shutdown(self, _sender: object, event: ShutdownRequested) -> None:
        """Stop renderer and show final status."""
        self._state.is_done = True
        self._state.phase = "Shutting down"

        # Stop billing for all instances
        now = time.monotonic()
        for inst in self._state.instances.values():
            if inst.end_time is None and inst.start_time is not None:
                inst.end_time = now

        # Stop renderer (includes grace period)
        self._renderer.stop()

        # Calculate and print final status
        total_cost, elapsed, savings = self._calculate_cost()
        self._renderer.print_final_status(
            has_error=self._state.has_error,
            total_cost=total_cost,
            elapsed=elapsed,
            savings=savings,
        )
        self._active = False

    # =========================================================================
    # Instance Lifecycle Events
    # =========================================================================

    @on(InstanceProvisioned)
    async def _on_instance_provisioned(self, _sender: object, event: InstanceProvisioned) -> None:
        """Track provisioned instance."""
        inst = event.instance

        # Remove placeholder
        placeholder = f"node-{self._state.provisioned}"
        self._state.instances.pop(placeholder, None)

        # Get pricing from InstanceInfo (populated by provider)
        self._state.instances[inst.id] = InstanceState(
            instance_id=inst.id,
            node=inst.node,
            provider=inst.provider,
            is_spot=inst.spot,
            hourly_rate=inst.hourly_rate,
            on_demand_rate=inst.on_demand_rate,
            billing_increment_minutes=inst.billing_increment,
            spec_name=inst.instance_type,
            start_time=time.monotonic(),
            metrics=MetricsState(),
        )

        self._state.provisioned += 1
        if inst.spot:
            self._state.spot_count += 1
        else:
            self._state.ondemand_count += 1

        # Capture infra from first instance (using InstanceInfo data)
        if not self._state.infra.provider:
            self._state.infra = InfraState(
                provider=inst.provider,
                region=self.spec.region,
                instance_type=inst.instance_type,
                vcpus=inst.vcpus,
                memory_gb=int(inst.memory_gb),
                gpu_count=inst.gpu_count,
                gpu_model=inst.gpu_model,
                gpu_vram_gb=inst.gpu_vram_gb,
            )

    @on(InstanceBootstrapped)
    async def _on_instance_bootstrapped(self, _sender: object, event: InstanceBootstrapped) -> None:
        """Mark instance as ready."""
        inst = self._state.instances.get(event.instance.id)
        if inst:
            inst.bootstrapped = True

        self._state.ready += 1
        total = self._state.total_nodes or self._state.provisioned

        if self._state.ready >= total > 0:
            self._state.phase = "Executing"

    @on(InstanceDestroyed)
    async def _on_instance_destroyed(self, _sender: object, event: InstanceDestroyed) -> None:
        """Mark instance end time for billing."""
        inst = self._state.instances.get(event.instance_id)
        if inst:
            inst.end_time = time.monotonic()

    @on(InstancePreempted)
    async def _on_preempted(self, _sender: object, event: InstancePreempted) -> None:
        """Handle spot instance preemption."""
        inst = self._state.instances.get(event.instance.id)
        if inst:
            inst.preempted = True
            self._add_log(inst, f"PREEMPTED: {event.reason}")

    # =========================================================================
    # Bootstrap Events
    # =========================================================================

    @on(BootstrapConsole, audit=False)
    async def _on_bootstrap_console(self, _sender: object, event: BootstrapConsole) -> None:
        """Add bootstrap console output to logs."""
        inst = self._state.instances.get(event.instance.id)
        if inst and event.content.strip():
            msg = event.content.strip()
            if not msg.startswith("#"):
                self._add_log(inst, msg[:80])

    @on(BootstrapPhase)
    async def _on_bootstrap_phase(self, _sender: object, event: BootstrapPhase) -> None:
        """Track bootstrap phase transitions."""
        self._state.phase = "Bootstrapping"
        inst = self._state.instances.get(event.instance.id)
        if not inst:
            return

        match event.event:
            case "started":
                self._add_log(inst, f"$ {event.phase}...")
            case "completed":
                elapsed = f" ({event.elapsed:.1f}s)" if event.elapsed else ""
                self._add_log(inst, f"  {event.phase} done{elapsed}")
            case "failed":
                self._add_log(inst, f"  {event.phase} FAILED: {event.error}")

    @on(BootstrapCommand, audit=False)
    async def _on_bootstrap_command(self, _sender: object, event: BootstrapCommand) -> None:
        """Show bootstrap command being executed."""
        inst = self._state.instances.get(event.instance.id)
        if inst:
            cmd = event.command.strip()
            if cmd:
                display_cmd = f"$ {cmd[:70]}..." if len(cmd) > 70 else f"$ {cmd}"
                self._add_log(inst, display_cmd)

    # =========================================================================
    # Metrics and Logs
    # =========================================================================

    @on(Metric, audit=False)  # High frequency, disable audit
    async def _on_metric(self, _sender: object, event: Metric) -> None:
        """Update metrics with EMA smoothing."""
        inst = self._state.instances.get(event.instance.id)
        if not inst:
            return

        metrics = inst.metrics

        # EMA smoothing (alpha=0.15)
        alpha = 0.15
        prev = metrics.smoothed.get(event.name, event.value)
        smoothed = alpha * event.value + (1 - alpha) * prev

        metrics.values[event.name] = smoothed
        metrics.smoothed[event.name] = smoothed

        if event.name not in metrics.history:
            metrics.history[event.name] = []
        metrics.history[event.name].append(smoothed)
        metrics.history[event.name] = metrics.history[event.name][-25:]

    @on(Log, audit=False)  # High frequency, disable audit
    async def _on_log(self, _sender: object, event: Log) -> None:
        """Append log line to instance."""
        inst = self._state.instances.get(event.instance.id)
        if inst:
            line = event.line.strip()
            if line:
                self._add_log(inst, line)

    # =========================================================================
    # Error Events
    # =========================================================================

    @on(Error)
    async def _on_error(self, _sender: object, event: Error) -> None:
        """Track error state."""
        self._state.has_error = True
        if event.fatal:
            self._state.phase = "Error"

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _add_log(self, inst: InstanceState, line: str) -> None:
        """Add a log line to an instance, handling in-place terminal updates.

        Keras and other tools use backspaces to update progress bars in-place.
        We detect these patterns and replace the last line instead of appending.
        """
        # Strip control characters (backspace, carriage return, newline)
        clean_line = line.replace("\x08", "").replace("\r", "").replace("\n", " ")

        # Skip empty lines (e.g., lines that were just backspaces)
        if not clean_line.strip():
            return

        # Detect progress bar updates (Keras-style with bar chars or step info)
        is_progress = any(c in clean_line for c in "━▁▂▃▄▅▆▇█") or "/step" in clean_line

        # If this is a progress update and last line was too, replace it
        if is_progress and inst.logs:
            last = inst.logs[-1]
            last_is_progress = any(c in last for c in "━▁▂▃▄▅▆▇█") or "/step" in last
            if last_is_progress:
                inst.logs[-1] = clean_line  # Replace in-place
                inst.last_log_time = time.monotonic()
                return

        # Normal append
        inst.logs.append(clean_line)
        inst.logs = inst.logs[-100:]  # Keep enough for large terminals
        inst.last_log_time = time.monotonic()

    def _calculate_cost(self) -> tuple[float, float, float]:
        """Calculate (total_cost, max_elapsed, savings)."""
        total_cost = 0.0
        total_ondemand = 0.0
        max_elapsed = 0.0

        for inst in self._state.instances.values():
            if inst.start_time is not None and not inst.is_placeholder:
                total_cost += inst.cost
                total_ondemand += inst.on_demand_cost
                max_elapsed = max(max_elapsed, inst.elapsed_seconds)

        savings = total_ondemand - total_cost
        return total_cost, max_elapsed, savings
