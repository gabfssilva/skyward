"""Panel state management.

Mutable state updated by events, with to_view_model() for rendering.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from math import ceil
from typing import Literal

from .components.metrics import temp_color
from .viewmodel import (
    ClusterVM,
    HeaderVM,
    InfraVM,
    InstanceVM,
    MetricVM,
    PanelViewModel,
)

# =============================================================================
# Per-Instance State
# =============================================================================


@dataclass(slots=True)
class MetricsState:
    """Per-instance metrics with history."""

    values: dict[str, float] = field(default_factory=dict)
    history: dict[str, list[float]] = field(default_factory=dict)
    smoothed: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class InstanceState:
    """State for a single instance."""

    instance_id: str
    node: int = 0  # Node index from event (0 = head)
    provider: str = ""
    spec_name: str = ""
    is_spot: bool = True
    hourly_rate: float = 0.0
    on_demand_rate: float = 0.0
    billing_increment_minutes: int | None = None
    start_time: float | None = None
    end_time: float | None = None
    logs: list[str] = field(default_factory=list)
    last_log_time: float = 0.0
    metrics: MetricsState = field(default_factory=MetricsState)
    preempted: bool = False  # Track preemption status
    bootstrapped: bool = False  # Track bootstrap completion

    @property
    def is_placeholder(self) -> bool:
        """Check if this is a placeholder instance (not yet provisioned)."""
        return self.instance_id.startswith("node-")

    @property
    def elapsed_seconds(self) -> float:
        """Calculate elapsed time since billing started."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.monotonic()
        return end - self.start_time

    @property
    def billable_hours(self) -> float:
        """Calculate billable hours, respecting billing increment."""
        if self.start_time is None:
            return 0.0

        elapsed_minutes = self.elapsed_seconds / 60

        if self.billing_increment_minutes is not None:
            increment = self.billing_increment_minutes
            billable_minutes = ceil(elapsed_minutes / increment) * increment
        else:
            billable_minutes = elapsed_minutes

        return billable_minutes / 60

    @property
    def cost(self) -> float:
        """Calculate current cost for this instance."""
        return self.billable_hours * self.hourly_rate

    @property
    def on_demand_cost(self) -> float:
        """Calculate what on-demand would cost (for savings)."""
        return self.billable_hours * self.on_demand_rate


# =============================================================================
# Infrastructure State
# =============================================================================


@dataclass(slots=True)
class InfraState:
    """Captured infrastructure info."""

    provider: str = ""
    region: str = ""
    instance_type: str = ""
    vcpus: int = 0
    memory_gb: int = 0
    gpu_count: int = 0
    gpu_model: str = ""
    gpu_vram_gb: int = 0
    allocation: str = ""  # "spot", "on-demand", "spot-if-available", "cheapest"
    dashboard_url: str | None = None


# =============================================================================
# Panel State
# =============================================================================


@dataclass(slots=True)
class PanelState:
    """Mutable state updated by events.

    This is the central state store that gets updated by the controller
    based on incoming events. It provides to_view_model() to produce
    immutable snapshots for rendering.
    """

    # Phase tracking
    phase: str = "Initializing"
    phase_times: dict[str, float] = field(default_factory=dict)
    start_time: float = 0.0
    is_done: bool = False
    has_error: bool = False

    # Node tracking
    total_nodes: int = 0
    provisioned: int = 0
    ready: int = 0
    spot_count: int = 0
    ondemand_count: int = 0

    # Infrastructure
    infra: InfraState = field(default_factory=InfraState)

    # Instances (instance_id -> InstanceState)
    instances: dict[str, InstanceState] = field(default_factory=dict)

    def to_view_model(
        self, terminal_width: int, terminal_height: int, blink_on: bool = True
    ) -> PanelViewModel:
        """Transform state to immutable ViewModel for rendering.

        Args:
            terminal_width: Current terminal width.
            terminal_height: Current terminal height.
            blink_on: Whether the blink indicator should be on.

        Returns:
            Immutable PanelViewModel ready for rendering.
        """
        return self._build_view_model(terminal_width, terminal_height, blink_on)

    def _build_view_model(self, width: int, height: int, blink_on: bool) -> PanelViewModel:
        """Build the complete ViewModel."""
        # Build header
        header = self._build_header_vm(blink_on)

        # Build infra
        infra = self._build_infra_vm()

        # Calculate log lines per instance type based on terminal height
        # Layout: header (1) + infra (3) + separator (1) + secondary nodes (1 header + 2 logs each) + head (1 header + N logs)
        active_id = self._get_active_instance_id()
        secondary_count = sum(
            1 for inst in self.instances.values()
            if not inst.is_placeholder and inst.instance_id != active_id
        )
        fixed_overhead = 1 + 3 + 1 + 1  # header + infra + separator + head header
        secondary_total = secondary_count * 3  # 1 header + 2 log lines each
        head_log_lines = max(5, height - fixed_overhead - secondary_total)

        # Build instance VMs with appropriate log slicing
        instance_vms = self._build_instance_vms(active_id, head_log_lines, blink_on)

        # Build cluster VM from instances
        cluster = self._build_cluster_vm(instance_vms)

        return PanelViewModel(
            header=header,
            infra=infra,
            cluster=cluster,
            instances=tuple(instance_vms),
            active_instance_id=active_id,
            terminal_width=width,
            terminal_height=height,
        )

    def _build_header_vm(self, blink_on: bool) -> HeaderVM:
        """Build header ViewModel."""
        elapsed = time.monotonic() - self.start_time if self.start_time else 0.0
        cost = sum(inst.cost for inst in self.instances.values() if not inst.is_placeholder)

        phases = self._build_pipeline_phases()

        return HeaderVM(
            cost=cost,
            elapsed_seconds=elapsed,
            phases=phases,
            blink_on=blink_on,
        )

    def _build_pipeline_phases(
        self,
    ) -> dict[str, Literal["pending", "in_progress", "completed"]]:
        """Build 3-phase pipeline: Provision - Prepare - Execute."""
        # Provision: covers network + instance provisioning
        if "provision" in self.phase_times:
            provision_status: Literal["pending", "in_progress", "completed"] = "completed"
        elif self.phase in ("Provisioning", "Initializing"):
            provision_status = "in_progress"
        else:
            provision_status = "pending"

        # Prepare: covers bootstrap
        if "bootstrap" in self.phase_times:
            prepare_status: Literal["pending", "in_progress", "completed"] = "completed"
        elif self.phase == "Bootstrapping":
            prepare_status = "in_progress"
        else:
            prepare_status = "pending"

        # Execute: running user code
        if self.is_done:
            execute_status: Literal["pending", "in_progress", "completed"] = "completed"
        elif self.phase == "Executing":
            execute_status = "in_progress"
        else:
            execute_status = "pending"

        return {
            "Provision": provision_status,
            "Prepare": prepare_status,
            "Execute": execute_status,
        }

    def _build_infra_vm(self) -> InfraVM | None:
        """Build infrastructure ViewModel."""
        if not self.infra.provider:
            return None

        # Build GPU info string: "1x T4 (16 GB)"
        gpu_info: str | None = None
        if self.infra.gpu_count and self.infra.gpu_model:
            vram_total = self.infra.gpu_count * self.infra.gpu_vram_gb
            if self.infra.gpu_vram_gb:
                gpu_info = f"{self.infra.gpu_count}x {self.infra.gpu_model} ({vram_total} GB)"
            else:
                gpu_info = f"{self.infra.gpu_count}x {self.infra.gpu_model}"

        # Build allocation string: "12 nodes (10 spot + 2 od)"
        total_nodes = self.total_nodes or self.provisioned
        if self.spot_count > 0 and self.ondemand_count > 0:
            allocation = f"{total_nodes} nodes ({self.spot_count} spot + {self.ondemand_count} od)"
        elif self.spot_count > 0:
            allocation = f"{total_nodes} spot"
        elif self.ondemand_count > 0:
            allocation = f"{total_nodes} on-demand"
        elif self.infra.allocation:
            # Use allocation strategy from spec before we have real data
            strategy = self.infra.allocation
            if strategy == "on-demand":
                allocation = f"{total_nodes} on-demand"
            elif strategy == "spot":
                allocation = f"{total_nodes} spot"
            else:
                # spot-if-available, cheapest
                allocation = f"{total_nodes} nodes"
        else:
            allocation = f"{total_nodes} nodes"

        # Calculate hourly rate
        total_hourly = sum(
            inst.hourly_rate for inst in self.instances.values() if not inst.is_placeholder
        )

        # Format hourly rate
        if total_hourly < 10:
            hourly_rate = f"${total_hourly:.2f}/hr"
        else:
            hourly_rate = f"${total_hourly:.0f}/hr"

        return InfraVM(
            provider=self.infra.provider,
            region=self.infra.region,
            instance_type=self.infra.instance_type,
            vcpus=self.infra.vcpus,
            memory_gb=self.infra.memory_gb,
            gpu_info=gpu_info,
            allocation=allocation,
            hourly_rate=hourly_rate,
            dashboard_url=self.infra.dashboard_url,
        )

    def _build_instance_vms(
        self, active_id: str | None, head_log_lines: int, blink_on: bool
    ) -> list[InstanceVM]:
        """Build InstanceVMs for all real (non-placeholder) instances.

        Args:
            active_id: ID of the head/active instance (gets more log lines).
            head_log_lines: Number of log lines for head node.
            blink_on: Whether blink indicator should be on (for status dot).
        """
        result: list[InstanceVM] = []
        secondary_log_lines = 2

        for inst_id, inst in self.instances.items():
            if inst.is_placeholder:
                continue

            # Extract metrics (labels padded to 4 chars for alignment)
            cpu = self._extract_metric(inst.metrics, "cpu", "cpu ", "%", "green")
            mem = self._extract_metric(inst.metrics, "mem", "mem ", "%", "cyan")
            gpu = self._extract_metric(inst.metrics, "gpu_util", "gpu ", "%", "magenta")
            gpu_mem = self._extract_gpu_mem_percent(inst.metrics)
            temp_raw = self._extract_metric(inst.metrics, "gpu_temp", "temp", "", "yellow")
            # Apply dynamic color based on temperature value
            temp = MetricVM(
                label=temp_raw.label,
                value=temp_raw.value,
                history=temp_raw.history,
                unit=temp_raw.unit,
                style=temp_color(temp_raw.value),
                spark_width=temp_raw.spark_width,
            )

            market: Literal["spot", "on-demand"] = "spot" if inst.is_spot else "on-demand"

            # Determine instance status
            status: Literal["bootstrapping", "ready", "done"]
            if self.is_done:
                status = "done"
            elif inst.bootstrapped:
                status = "ready"
            else:
                status = "bootstrapping"

            # Slice logs based on instance type
            is_active = inst_id == active_id
            log_count = head_log_lines if is_active else secondary_log_lines
            logs = tuple(inst.logs[-log_count:]) if inst.logs else ()

            result.append(
                InstanceVM(
                    instance_id=inst_id,
                    short_id=inst_id[:10] if len(inst_id) > 10 else inst_id,
                    market=market,
                    status=status,
                    cpu=cpu,
                    mem=mem,
                    gpu=gpu,
                    gpu_mem=gpu_mem,
                    temp=temp,
                    logs=logs,
                    is_active=is_active,
                    blink_on=blink_on,
                )
            )

        return result

    def _extract_metric(
        self,
        metrics: MetricsState,
        base_name: str,
        label: str,
        unit: str,
        style: str,
    ) -> MetricVM:
        """Extract metric history and current value, aggregating multi-GPU if needed."""
        # Find all raw metric names that match this base (e.g., gpu_util_0, gpu_util_1)
        matching_names = [
            name
            for name in metrics.history
            if name == base_name or re.match(rf"^{re.escape(base_name)}_\d+$", name)
        ]

        if not matching_names:
            return MetricVM(
                label=label,
                value=0.0,
                history=(),
                unit=unit,
                style=style,
                spark_width=8,
            )

        if len(matching_names) == 1:
            # Single metric
            history = metrics.history.get(matching_names[0], [])
            current = history[-1] if history else 0.0
            return MetricVM(
                label=label,
                value=current,
                history=tuple(history),
                unit=unit,
                style=style,
                spark_width=8,
            )

        # Multiple metrics - average them
        histories = [metrics.history.get(n, []) for n in matching_names]
        max_len = max((len(h) for h in histories), default=0)

        avg_history: list[float] = []
        for idx in range(max_len):
            vals = [h[idx] for h in histories if idx < len(h)]
            avg_history.append(sum(vals) / len(vals) if vals else 0.0)

        current = avg_history[-1] if avg_history else 0.0
        return MetricVM(
            label=label,
            value=current,
            history=tuple(avg_history),
            unit=unit,
            style=style,
            spark_width=8,
        )

    def _extract_gpu_mem_percent(self, metrics: MetricsState) -> MetricVM:
        """Extract GPU memory usage as percentage.

        Calculates percentage from gpu_mem_mb and gpu_mem_total_mb metrics.
        """
        # Find used and total metrics (may have per-GPU suffixes)
        used_names = [
            n for n in metrics.history
            if n == "gpu_mem_mb" or re.match(r"^gpu_mem_mb_\d+$", n)
        ]
        total_names = [
            n for n in metrics.history
            if n == "gpu_mem_total_mb" or re.match(r"^gpu_mem_total_mb_\d+$", n)
        ]

        if not used_names:
            return MetricVM(
                label="vram", value=0.0, history=(), unit="%", style="blue", spark_width=8
            )

        # Get the total (usually constant, take first value)
        total_mb = 0.0
        for name in total_names:
            hist = metrics.history.get(name, [])
            if hist:
                total_mb = hist[0]
                break

        if total_mb == 0:
            # Fallback: no total available
            return MetricVM(
                label="vram", value=0.0, history=(), unit="%", style="blue", spark_width=8
            )

        # Calculate percentage for each history point
        used_histories = [metrics.history.get(n, []) for n in used_names]
        max_len = max((len(h) for h in used_histories), default=0)

        pct_history: list[float] = []
        for idx in range(max_len):
            vals = [h[idx] for h in used_histories if idx < len(h)]
            avg_used = sum(vals) / len(vals) if vals else 0.0
            pct = (avg_used / total_mb) * 100
            pct_history.append(min(pct, 100.0))

        current = pct_history[-1] if pct_history else 0.0

        return MetricVM(
            label="vram",
            value=current,
            history=tuple(pct_history),
            unit="%",
            style="blue",
            spark_width=8,
        )

    def _get_active_instance_id(self) -> str | None:
        """Get ID of head instance (node 0)."""
        for inst_id, inst in self.instances.items():
            if not inst.is_placeholder and inst.node == 0:
                return inst_id
        return None

    def _build_cluster_vm(self, instances: list[InstanceVM]) -> ClusterVM:
        """Calculate cluster-wide average metrics."""
        if not instances:
            return ClusterVM(
                cpu=MetricVM(
                    label="cpu ", value=0.0, history=(), unit="%", style="green", spark_width=8
                ),
                mem=MetricVM(
                    label="mem ", value=0.0, history=(), unit="%", style="cyan", spark_width=8
                ),
                gpu=MetricVM(
                    label="gpu ", value=0.0, history=(), unit="%", style="magenta", spark_width=8
                ),
                gpu_mem=MetricVM(
                    label="vram", value=0.0, history=(), unit="%", style="blue", spark_width=8
                ),
                temp=MetricVM(
                    label="temp", value=0.0, history=(), unit="", style="green", spark_width=8
                ),
            )

        # Average current values
        avg_cpu = sum(i.cpu.value for i in instances) / len(instances)
        avg_mem = sum(i.mem.value for i in instances) / len(instances)
        avg_gpu = sum(i.gpu.value for i in instances) / len(instances)
        avg_gpu_mem = sum(i.gpu_mem.value for i in instances) / len(instances)
        avg_temp = sum(i.temp.value for i in instances) / len(instances)

        # Average histories
        cpu_history = self._avg_histories([i.cpu.history for i in instances])
        mem_history = self._avg_histories([i.mem.history for i in instances])
        gpu_history = self._avg_histories([i.gpu.history for i in instances])
        gpu_mem_history = self._avg_histories([i.gpu_mem.history for i in instances])
        temp_history = self._avg_histories([i.temp.history for i in instances])

        return ClusterVM(
            cpu=MetricVM(
                label="cpu ",
                value=avg_cpu,
                history=cpu_history,
                unit="%",
                style="green",
                spark_width=8,
            ),
            mem=MetricVM(
                label="mem ",
                value=avg_mem,
                history=mem_history,
                unit="%",
                style="cyan",
                spark_width=8,
            ),
            gpu=MetricVM(
                label="gpu ",
                value=avg_gpu,
                history=gpu_history,
                unit="%",
                style="magenta",
                spark_width=8,
            ),
            gpu_mem=MetricVM(
                label="vram",
                value=avg_gpu_mem,
                history=gpu_mem_history,
                unit="%",
                style="blue",
                spark_width=8,
            ),
            temp=MetricVM(
                label="temp",
                value=avg_temp,
                history=temp_history,
                unit="",
                style=temp_color(avg_temp),
                spark_width=8,
            ),
        )

    def _avg_histories(self, histories: list[tuple[float, ...]]) -> tuple[float, ...]:
        """Average multiple history tuples aligned by index."""
        if not histories:
            return ()

        max_len = max((len(h) for h in histories), default=0)
        if max_len == 0:
            return ()

        result: list[float] = []
        for idx in range(max_len):
            vals = [h[idx] for h in histories if idx < len(h)]
            result.append(sum(vals) / len(vals) if vals else 0.0)

        return tuple(result)
