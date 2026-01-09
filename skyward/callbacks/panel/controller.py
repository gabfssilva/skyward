"""Panel controller.

Thin event dispatcher that updates state and triggers rendering.
"""

from __future__ import annotations

import time
from functools import singledispatchmethod

from skyward.core.events import (
    BootstrapCompleted,
    BootstrapProgress,
    BootstrapStarting,
    Error,
    FunctionCall,
    FunctionResult,
    InstanceLaunching,
    InstanceProvisioned,
    InstanceReady,
    InstanceStopping,
    LogLine,
    MetricValue,
    NetworkReady,
    PoolReady,
    PoolStarted,
    PoolStopping,
    ProvisioningCompleted,
    ProvisioningStarted,
    SkywardEvent,
)

from .renderer import PanelRenderer
from .state import InfraState, InstanceState, MetricsState, PanelState


def _short_id(instance_id: str) -> str:
    """Shorten instance ID for display."""
    return instance_id[:12] if len(instance_id) > 12 else instance_id


class PanelController:
    """Thin event dispatcher that updates state and triggers rendering.

    Responsibilities:
    - Handle SkywardEvents via singledispatch
    - Update PanelState based on events
    - Rich Live auto-refreshes at configured rate
    """

    def __init__(self, renderer: PanelRenderer) -> None:
        self._state = PanelState()
        self._renderer = renderer

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

    @singledispatchmethod
    def handle(self, event: SkywardEvent) -> None:
        """Handle unknown events - ignore."""

    @handle.register
    def _(self, event: PoolStarted) -> None:
        self._state.start_time = time.monotonic()
        self._state.total_nodes = event.nodes
        self._state.is_done = False
        self._state.has_error = False
        self._state.instances.clear()

        # Initialize placeholders
        for i in range(event.nodes):
            self._state.instances[f"node-{i}"] = InstanceState(f"node-{i}")

        self._renderer.start(self._state)

    @handle.register
    def _(self, event: PoolStopping) -> None:
        self._state.is_done = True

        # Stop billing for all instances
        now = time.monotonic()
        for inst in self._state.instances.values():
            if inst.end_time is None and inst.start_time is not None:
                inst.end_time = now

        # Stop renderer (includes grace period for final logs)
        self._renderer.stop()

        total_cost, elapsed, savings = self._calculate_cost()
        self._renderer.print_final_status(
            has_error=self._state.has_error,
            total_cost=total_cost,
            elapsed=elapsed,
            savings=savings,
        )

    @handle.register
    def _(self, event: ProvisioningStarted) -> None:
        self._state.phase = "Provisioning"

    @handle.register
    def _(self, event: NetworkReady) -> None:
        self._state.infra.region = event.region
        elapsed = time.monotonic() - self._state.start_time if self._state.start_time else 0.0
        self._state.phase_times["network"] = elapsed

    @handle.register
    def _(self, event: InstanceLaunching) -> None:
        if event.count:
            self._state.total_nodes = event.count

    @handle.register
    def _(self, event: InstanceProvisioned) -> None:
        inst = event.instance

        # Remove placeholder
        placeholder = f"node-{self._state.provisioned}"
        self._state.instances.pop(placeholder, None)

        # Get pricing from spec
        if inst.spec:
            hourly = inst.spec.price_spot if inst.spot else inst.spec.price_on_demand
            on_demand = inst.spec.price_on_demand
            billing_increment = inst.spec.billing_increment_minutes
        else:
            hourly = 0.0
            on_demand = 0.0
            billing_increment = None

        # Add real instance
        self._state.instances[inst.instance_id] = InstanceState(
            instance_id=inst.instance_id,
            node=inst.node,
            provider=inst.provider.value if inst.provider else "",
            spec_name=inst.spec.name if inst.spec else "",
            is_spot=inst.spot,
            hourly_rate=hourly or 0.0,
            on_demand_rate=on_demand or 0.0,
            billing_increment_minutes=billing_increment,
            start_time=time.monotonic(),
            metrics=MetricsState(),
        )

        self._state.provisioned += 1
        if inst.spot:
            self._state.spot_count += 1
        else:
            self._state.ondemand_count += 1

        # Capture infra from first instance
        if not self._state.infra.provider and inst.spec:
            self._state.infra = InfraState(
                provider=inst.provider.value.lower() if inst.provider else "",
                region=self._state.infra.region,  # Keep existing region
                instance_type=inst.spec.name,
                vcpus=inst.spec.vcpu or 0,
                memory_gb=int(inst.spec.memory_gb or 0),
                gpu_count=int(inst.spec.accelerator_count or 0),
                gpu_model=inst.spec.accelerator or "",
                gpu_vram_gb=int(inst.spec.accelerator_memory_gb or 0),
            )

    @handle.register
    def _(self, event: ProvisioningCompleted) -> None:
        self._state.phase = "Bootstrapping"
        self._state.infra.region = event.region
        elapsed = time.monotonic() - self._state.start_time if self._state.start_time else 0.0
        self._state.phase_times["provision"] = elapsed

    @handle.register
    def _(self, event: BootstrapStarting) -> None:
        self._state.phase = "Bootstrapping"

    @handle.register
    def _(self, event: BootstrapProgress) -> None:
        self._state.phase = "Bootstrapping"
        full_inst_id = event.instance.instance_id
        inst = self._state.instances.get(full_inst_id)

        if inst:
            # Add log lines
            if event.step == "command" and event.message:
                cmd = event.message.strip()
                if cmd:
                    display_cmd = f"$ {cmd[:70]}..." if len(cmd) > 70 else f"$ {cmd}"
                    self._add_log(inst, display_cmd)
            elif event.step == "console" and event.message:
                msg = event.message.strip()
                if msg and not msg.startswith("#"):
                    self._add_log(inst, msg[:80])

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

        # Detect progress bar updates (Keras-style with ━ or step info)
        is_progress = "━" in clean_line or "/step" in clean_line

        # If this is a progress update and last line was too, replace it
        if is_progress and inst.logs:
            last = inst.logs[-1]
            last_is_progress = "━" in last or "/step" in last
            if last_is_progress:
                inst.logs[-1] = clean_line  # Replace in-place
                inst.last_log_time = time.monotonic()
                return

        # Normal append
        inst.logs.append(clean_line)
        inst.logs = inst.logs[-100:]  # Keep enough for large terminals
        inst.last_log_time = time.monotonic()

    @handle.register
    def _(self, event: BootstrapCompleted) -> None:
        self._state.ready += 1
        total = self._state.total_nodes or self._state.provisioned

        if self._state.ready >= total > 0:
            self._state.phase = "Executing"

    @handle.register
    def _(self, event: InstanceReady) -> None:
        pass

    @handle.register
    def _(self, event: PoolReady) -> None:
        self._state.phase = "Executing"
        self._state.ready = len(event.instances)
        elapsed = time.monotonic() - self._state.start_time if self._state.start_time else 0.0
        self._state.phase_times["bootstrap"] = elapsed

    @handle.register
    def _(self, event: MetricValue) -> None:
        inst = self._state.instances.get(event.instance.instance_id)
        if inst:
            metrics = inst.metrics

            # EMA smoothing
            alpha = 0.15
            prev = metrics.smoothed.get(event.name, event.value)
            smoothed = alpha * event.value + (1 - alpha) * prev

            metrics.values[event.name] = smoothed
            metrics.smoothed[event.name] = smoothed

            if event.name not in metrics.history:
                metrics.history[event.name] = []
            metrics.history[event.name].append(smoothed)
            metrics.history[event.name] = metrics.history[event.name][-25:]

    @handle.register
    def _(self, event: LogLine) -> None:
        inst = self._state.instances.get(event.instance.instance_id)
        if inst:
            line = event.line.strip()
            if line:
                self._add_log(inst, line)

    @handle.register
    def _(self, event: FunctionCall) -> None:
        pass

    @handle.register
    def _(self, event: FunctionResult) -> None:
        pass

    @handle.register
    def _(self, event: InstanceStopping) -> None:
        self._state.phase = "Shutting down"
        inst = self._state.instances.get(event.instance.instance_id)
        if inst:
            inst.end_time = time.monotonic()

    @handle.register
    def _(self, event: Error) -> None:
        self._state.has_error = True
        self._state.phase = "Error"
