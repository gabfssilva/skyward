"""Rich-based panel callback for beautiful terminal UI."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import singledispatchmethod
from math import ceil
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Literal

from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.style import Style
from rich.table import Table
from rich.text import Text

from skyward.callbacks._panel_components import (
    InfraSpec,
    InstanceMetrics,
    create_header,
    create_infra_section,
    create_panel,
)
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
    Metrics,
    NetworkReady,
    PoolReady,
    PoolStarted,
    PoolStopping,
    ProvisioningCompleted,
    ProvisioningStarted,
    SkywardEvent,
)

if TYPE_CHECKING:
    from skyward.core.callback import Callback

# Style constants
STYLE_DIM = Style(color="bright_black")
STYLE_PHASE = Style(color="cyan", bold=True)
STYLE_COST = Style(color="green")
STYLE_SPOT = Style(color="green")
STYLE_ONDEMAND = Style(color="blue")
STYLE_SUCCESS = Style(color="green")
STYLE_ERROR = Style(color="red", bold=True)
STYLE_INFO = Style(color="cyan")

# Thresholds (low, high)
_CPU_THRESHOLDS = (60.0, 85.0)
_GPU_THRESHOLDS = (70.0, 90.0)
_MEM_THRESHOLDS = (70.0, 90.0)

# Number of metric rows (Nodes, CPU, GPU, Mem, GMem)
_METRIC_ROWS = 5


def _short_id(instance_id: str) -> str:
    return instance_id[:12] if len(instance_id) > 12 else instance_id


def _format_duration(seconds: float) -> str:
    mins, secs = divmod(int(seconds), 60)
    if mins > 0:
        return f"{mins}m{secs:02d}s"
    return f"{secs}s"


def _metric_color(value: float, thresholds: tuple[float, float]) -> str:
    """Return color based on thresholds (low=green, mid=yellow, high=red)."""
    low, high = thresholds
    if value < low:
        return "green"
    if value < high:
        return "yellow"
    return "red"


def _metric_color_inverted(value: float, thresholds: tuple[float, float]) -> str:
    """Inverted: high=green (good utilization), low=yellow/red."""
    low, high = thresholds
    if value >= high:
        return "green"
    if value >= low:
        return "yellow"
    return "red"


@dataclass(slots=True)
class _NodeMetrics:
    """Metrics for a single node."""

    cpu: float | None = None
    gpu: float | None = None
    mem: float | None = None
    gpu_mem: float | None = None
    gpu_temp: float | None = None


@dataclass(slots=True)
class _InstanceCost:
    """Tracks cost for a single instance."""

    instance_id: str
    hourly_rate: float  # Price we're paying (spot or on-demand)
    on_demand_rate: float  # On-demand price for savings calculation
    billing_increment_minutes: int | None = None  # None = per-second, 1 = AWS, 10 = Verda
    start_time: float | None = None
    end_time: float | None = None

    @property
    def elapsed_seconds(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.monotonic()
        return end - self.start_time

    @property
    def billable_hours(self) -> float:
        """Calculate billable hours, respecting billing increment.

        When billing starts, immediately counts the first block.
        AWS: 1 min minimum, Verda: 10 min minimum.
        """
        if self.start_time is None:
            return 0.0

        elapsed_minutes = self.elapsed_seconds / 60

        if self.billing_increment_minutes is not None:
            increment = self.billing_increment_minutes
            # ceil(0.001/10) = 1, so 1 * 10 = 10 min (first block)
            billable_minutes = ceil(elapsed_minutes / increment) * increment
        else:
            # Per-second billing
            billable_minutes = elapsed_minutes

        return billable_minutes / 60

    @property
    def cost(self) -> float:
        return self.billable_hours * self.hourly_rate

    @property
    def on_demand_cost(self) -> float:
        return self.billable_hours * self.on_demand_rate


@dataclass(slots=True)
class _Tracking:
    """State tracked across events."""

    phase: str = "Initializing"
    action: str = ""
    total_nodes: int = 0
    provisioned: int = 0
    ready: int = 0
    spot: int = 0
    ondemand: int = 0
    node_metrics: dict[int, _NodeMetrics] = field(default_factory=dict)
    instance_types: set[str] = field(default_factory=set)
    milestones: list[Text] = field(default_factory=list)
    log_lines: list[Text] = field(default_factory=list)
    has_error: bool = False

    # New fields for enhanced panel
    # instance_id -> [(cpu, gpu, vram, mem, temp), ...]
    metrics_history: dict[str, list[tuple[float, float, float, float, float]]] = field(
        default_factory=dict
    )
    # instance_id -> (cpu, gpu, vram, mem, temp) smoothed for EMA
    smoothed_metrics: dict[str, tuple[float, float, float, float, float]] = field(
        default_factory=dict
    )
    phase_times: dict[str, float] = field(default_factory=dict)  # phase_name -> elapsed
    recent_calls: list[str] = field(default_factory=list)  # last 3 function names
    infra_spec: InfraSpec | None = None
    infra_region: str = ""  # from ProvisioningCompleted
    start_time: float = 0.0  # pool start time for elapsed calculation
    execute_start_time: float = 0.0  # set at PoolReady for execute phase timing


class PanelController:
    """Controls the Rich Live display with 2-column layout."""

    _MAX_LOG_LINES = 50
    _MAX_VISIBLE_NODES = 4
    _PAGE_ROTATION_SECONDS = 3.0
    _SPINNER_FRAMES = ("🟢", "⚪")

    def __init__(self) -> None:
        self._console = Console()
        self._live: Live | None = None
        self._layout: Layout | None = None
        self._tracking = _Tracking()
        self._lock = Lock()
        self._is_done = False
        self._instance_costs: dict[str, _InstanceCost] = {}
        # Background refresh thread for animations
        self._refresh_thread: Thread | None = None
        self._stop_refresh = Event()
        # Pagination state
        self._current_page = 0
        self._last_page_change = 0.0

    def _calculate_panel_height(self) -> int:
        """Calculate panel height based on number of visible instances.

        Left panel is always 7 rows (header, separator, infra×5).
        Right panel: header(1) + visible_instances + (empty + pagination + average if >1 total).
        When paginating, always use MAX_VISIBLE_NODES to keep average position fixed.
        """
        num_instances = len(self._instance_costs)
        # When paginating, use full MAX_VISIBLE_NODES for consistent height
        is_paginating = num_instances > self._MAX_VISIBLE_NODES
        visible = self._MAX_VISIBLE_NODES if is_paginating else num_instances
        # Right panel rows: header(1) + visible + empty(1) + average(1) if total > 1
        right_rows = 2 if num_instances <= 1 else 1 + visible + 3
        # Left panel is 7 rows
        return max(7, right_rows)

    def _add_milestone(self, text: Text) -> None:
        """Add a milestone to the log, keeping max count."""
        self._tracking.milestones.append(text)
        # Keep only enough to fill the right column
        if len(self._tracking.milestones) > _METRIC_ROWS:
            self._tracking.milestones.pop(0)

    def _add_log_line(self, text: Text) -> None:
        """Add a log line (stdout from nodes), shown outside panel."""
        self._tracking.log_lines.append(text)
        if len(self._tracking.log_lines) > self._MAX_LOG_LINES:
            self._tracking.log_lines.pop(0)

    def _get_spinner_emoji(self) -> str:
        """Get current spinner emoji, alternating 🟢 ⚪ every 0.25s."""
        if self._is_done:
            return "🟢"
        # Alternate based on time (4 cycles per second = every 0.25s)
        idx = int(time.monotonic() * 4) % 2
        return self._SPINNER_FRAMES[idx]

    def _calculate_cost(self) -> tuple[float, float, float]:
        """Calculate (total_cost, elapsed_seconds, savings).

        Returns:
            Tuple of (total_cost, max_elapsed_seconds, savings_vs_ondemand).
        """
        if not self._instance_costs:
            return 0.0, 0.0, 0.0

        total_cost = 0.0
        total_ondemand = 0.0
        max_elapsed = 0.0

        for inst in self._instance_costs.values():
            if inst.start_time is not None:
                total_cost += inst.cost
                total_ondemand += inst.on_demand_cost
                max_elapsed = max(max_elapsed, inst.elapsed_seconds)

        savings = total_ondemand - total_cost
        return total_cost, max_elapsed, savings

    def _avg_metrics(self) -> _NodeMetrics:
        """Calculate average metrics across all nodes."""
        metrics = self._tracking.node_metrics
        if not metrics:
            return _NodeMetrics()

        cpu_vals = [m.cpu for m in metrics.values() if m.cpu is not None]
        gpu_vals = [m.gpu for m in metrics.values() if m.gpu is not None]
        mem_vals = [m.mem for m in metrics.values() if m.mem is not None]
        gpu_mem_vals = [m.gpu_mem for m in metrics.values() if m.gpu_mem is not None]

        return _NodeMetrics(
            cpu=sum(cpu_vals) / len(cpu_vals) if cpu_vals else None,
            gpu=sum(gpu_vals) / len(gpu_vals) if gpu_vals else None,
            mem=sum(mem_vals) / len(mem_vals) if mem_vals else None,
            gpu_mem=sum(gpu_mem_vals) / len(gpu_mem_vals) if gpu_mem_vals else None,
        )

    def _build_pipeline_phases(
        self,
    ) -> dict[str, Literal["pending", "in_progress", "completed"]]:
        """Build 3-phase pipeline: Provision → Prepare → Execute."""
        t = self._tracking

        # Provision: covers network + instance provisioning
        provision_status: Literal["pending", "in_progress", "completed"]
        if "provision" in t.phase_times:
            provision_status = "completed"
        elif t.phase in ("Provisioning", "Initializing"):
            provision_status = "in_progress"
        else:
            provision_status = "pending"

        # Prepare: covers bootstrap
        prepare_status: Literal["pending", "in_progress", "completed"]
        if "bootstrap" in t.phase_times:
            prepare_status = "completed"
        elif t.phase == "Bootstrapping":
            prepare_status = "in_progress"
        else:
            prepare_status = "pending"

        # Execute: running user code
        execute_status: Literal["pending", "in_progress", "completed"]
        if self._is_done:
            execute_status = "completed"
        elif t.phase == "Executing":
            execute_status = "in_progress"
        else:
            execute_status = "pending"

        return {
            "Provision": provision_status,
            "Prepare": prepare_status,
            "Execute": execute_status,
        }

    def _build_instance_metrics(self) -> tuple[list[InstanceMetrics], list[InstanceMetrics]]:
        """Build instance metrics list for metrics table.

        Returns:
            Tuple of (visible_instances, all_instances) where visible_instances
            is the current page and all_instances is used for cluster average.
        """
        t = self._tracking
        all_instances = []

        # Iterate over ALL provisioned instances, not just those with metrics
        for inst_id, cost_info in self._instance_costs.items():
            history = t.metrics_history.get(inst_id, [])

            cpu_history = tuple(h[0] for h in history) if history else ()
            gpu_history = tuple(h[1] for h in history) if history else ()
            vram_history = tuple(h[2] for h in history) if history else ()
            mem_history = tuple(h[3] for h in history) if history else ()
            temp_history = tuple(h[4] for h in history) if history else ()

            cpu_current = history[-1][0] if history else 0.0
            gpu_current = history[-1][1] if history else 0.0
            vram_current = history[-1][2] if history else 0.0
            mem_current = history[-1][3] if history else 0.0
            temp_current = history[-1][4] if history else 0.0

            # Determine market type from cost rates
            market = "on-demand" if cost_info.hourly_rate == cost_info.on_demand_rate else "spot"

            all_instances.append(InstanceMetrics(
                instance_id=inst_id,
                cpu_history=cpu_history,
                gpu_history=gpu_history,
                vram_history=vram_history,
                mem_history=mem_history,
                temp_history=temp_history,
                cpu_current=cpu_current,
                gpu_current=gpu_current,
                vram_current=vram_current,
                mem_current=mem_current,
                temp_current=temp_current,
                market=market,  # type: ignore
            ))

        # Paginate: return only current page of instances
        start_idx = self._current_page * self._MAX_VISIBLE_NODES
        end_idx = start_idx + self._MAX_VISIBLE_NODES
        visible_instances = all_instances[start_idx:end_idx]

        return visible_instances, all_instances

    def _build_panel(self) -> Table:
        """Build the main status panel with side-by-side layout."""
        t = self._tracking

        # Calculate current cost and elapsed
        elapsed = time.monotonic() - t.start_time if t.start_time else 0.0
        cost, _, _ = self._calculate_cost()
        blink_on = int(time.monotonic() * 2) % 2 == 0
        tick = int(time.monotonic() * 10)  # Spinner at ~10 FPS

        # Pipeline phases for header
        phases = self._build_pipeline_phases()

        # Header (includes status and spinner)
        header = create_header(
            cost,
            elapsed,
            phases,
            tick=tick if not self._is_done else 0,
            blink_on=blink_on and not self._is_done,
        )

        # Infra section (3 lines)
        if t.infra_spec:
            total_hourly = sum(
                c.hourly_rate for c in self._instance_costs.values()
            )
            total_nodes = t.total_nodes or t.provisioned
            # Calculate average price per node for correct display math
            avg_price_per_node = total_hourly / total_nodes if total_nodes > 0 else 0
            infra = create_infra_section(InfraSpec(
                provider=t.infra_spec.provider,
                region=t.infra_spec.region,
                instance_type=t.infra_spec.instance_type,
                vcpus=t.infra_spec.vcpus,
                memory_gb=t.infra_spec.memory_gb,
                gpu_count=t.infra_spec.gpu_count,
                gpu_model=t.infra_spec.gpu_model,
                gpu_vram_gb=t.infra_spec.gpu_vram_gb,
                total_nodes=total_nodes,
                spot_count=t.spot,
                ondemand_count=t.ondemand,
                hourly_rate=total_hourly,
                price_per_node=avg_price_per_node,
            ))
        else:
            # Fallback during early provisioning (5 lines)
            infra = (
                Text(f"{t.action or t.phase}", style="dim"),
                Text(""),
                Text(""),
                Text(""),
                Text(""),
            )

        # Instance metrics (paginated)
        visible_instances, all_instances = self._build_instance_metrics()

        # Assemble panel
        return create_panel(
            header=header,
            infra=infra,
            visible_instances=visible_instances,
            all_instances=all_instances,
            max_visible=self._MAX_VISIBLE_NODES,
            terminal_width=self._console.size.width,
        )

    def _build_display(self) -> Layout:
        """Build complete display using Layout for fixed panel + scrolling logs."""
        if self._layout is None:
            return Layout()  # Should not happen

        # Update panel height dynamically based on instance count
        panel_height = self._calculate_panel_height()
        self._layout["panel"].size = panel_height

        # Update panel section (centered)
        panel_table = self._build_panel()
        self._layout["panel"].update(Align.center(panel_table))

        # Update logs section with separator
        logs_content = Text()
        logs_content.append("─" * self._console.size.width, style="dim")

        if self._tracking.log_lines:
            logs_height = max(1, self._console.size.height - panel_height - 2)
            visible_logs = self._tracking.log_lines[-logs_height:]
            for log in visible_logs:
                logs_content.append("\n")
                logs_content.append_text(log)
            self._layout["logs"].update(logs_content)
        else:
            self._layout["logs"].update(logs_content)

        return self._layout

    def _refresh(self) -> None:
        """Refresh the live display."""
        if self._live:
            self._live.update(self._build_display())

    def _refresh_loop(self) -> None:
        """Background thread that refreshes display 4x/sec for animations."""
        while not self._stop_refresh.wait(0.25):
            with self._lock:
                if self._live and not self._is_done:
                    # Auto-rotate pages when there are more instances than fit on screen
                    self._maybe_rotate_page()
                    self._live.update(self._build_display())

    def _maybe_rotate_page(self) -> None:
        """Rotate to next page if enough time has passed."""
        num_instances = len(self._instance_costs)
        if num_instances <= self._MAX_VISIBLE_NODES:
            self._current_page = 0
            return

        now = time.monotonic()
        if now - self._last_page_change >= self._PAGE_ROTATION_SECONDS:
            total_pages = ceil(num_instances / self._MAX_VISIBLE_NODES)
            self._current_page = (self._current_page + 1) % total_pages
            self._last_page_change = now

    @singledispatchmethod
    def handle(self, event: SkywardEvent) -> None:
        """Handle unknown events - ignore."""

    @handle.register
    def _(self, event: PoolStarted) -> None:
        with self._lock:
            self._tracking = _Tracking()
            self._tracking.start_time = time.monotonic()
            self._tracking.total_nodes = event.nodes
            self._instance_costs = {}
            self._is_done = False
            # Reset pagination state
            self._current_page = 0
            self._last_page_change = time.monotonic()

            # Create placeholder entries for nodes (will be updated in InstanceProvisioned)
            for i in range(event.nodes):
                placeholder_id = f"node-{i}"
                self._instance_costs[placeholder_id] = _InstanceCost(
                    instance_id=placeholder_id,
                    hourly_rate=0.0,
                    on_demand_rate=0.0,
                )

            # Create layout with panel on top (dynamic height), logs below
            self._layout = Layout()
            self._layout.split_column(
                Layout(name="panel", size=self._calculate_panel_height()),
                Layout(name="logs"),
            )

            # Start Live display
            self._live = Live(
                self._layout,
                console=self._console,
                refresh_per_second=4,
                transient=False,
            )
            self._live.start()

            # Start background refresh thread for animations
            self._stop_refresh.clear()
            self._refresh_thread = Thread(target=self._refresh_loop, daemon=True)
            self._refresh_thread.start()

    @handle.register
    def _(self, event: PoolStopping) -> None:
        with self._lock:
            self._is_done = True
            self._tracking.action = "Complete"

            # Stop all billing
            now = time.monotonic()
            for inst in self._instance_costs.values():
                if inst.end_time is None and inst.start_time is not None:
                    inst.end_time = now

            self._refresh()

        # Stop refresh thread (outside lock to avoid deadlock)
        self._stop_refresh.set()
        if self._refresh_thread:
            self._refresh_thread.join(timeout=1.0)
            self._refresh_thread = None

        with self._lock:
            if self._live:
                self._live.stop()
                self._live = None

                # Calculate final cost
                total_cost, elapsed, savings = self._calculate_cost()

                # Print final status
                t = self._tracking
                if t.has_error:
                    self._console.print("[bold red]✗[/bold red] Pool stopped with errors")
                else:
                    final = Text()
                    final.append("✓ ", style=STYLE_SUCCESS)
                    final.append("Complete", style="bold green")
                    if total_cost > 0:
                        final.append(f" │ Cost: ${total_cost:.4f}", style=STYLE_DIM)
                    if elapsed > 0:
                        final.append(f" │ Duration: {_format_duration(elapsed)}", style=STYLE_DIM)
                    if savings > 0:
                        final.append(f" │ Saved: ${savings:.4f}", style=STYLE_SPOT)
                    self._console.print(final)

    @handle.register
    def _(self, event: ProvisioningStarted) -> None:
        with self._lock:
            self._tracking.phase = "Provisioning"
            self._tracking.action = "Starting provisioning..."
            self._refresh()

    @handle.register
    def _(self, event: NetworkReady) -> None:
        with self._lock:
            t = self._tracking
            t.action = f"Network ready ({event.region})"
            t.infra_region = event.region

            # Track phase time
            elapsed = time.monotonic() - t.start_time if t.start_time else 0.0
            t.phase_times["network"] = elapsed

            milestone = Text()
            milestone.append("✓ ", style=STYLE_SUCCESS)
            milestone.append(f"Network ready in {event.region}", style=STYLE_DIM)
            self._add_milestone(milestone)

            self._refresh()

    @handle.register
    def _(self, event: InstanceLaunching) -> None:
        with self._lock:
            if event.count:
                self._tracking.total_nodes = event.count

            candidates = [c.name for c in event.candidates] if event.candidates else []
            if candidates:
                main = candidates[0]
                suffix = f" (+{len(candidates) - 1})" if len(candidates) > 1 else ""
                self._tracking.action = f"Launching {event.count} × {main}{suffix}"
            else:
                self._tracking.action = f"Launching {event.count} instances..."

            self._refresh()

    @handle.register
    def _(self, event: InstanceProvisioned) -> None:
        with self._lock:
            t = self._tracking
            inst = event.instance

            # Remove a placeholder if exists (replace with real instance)
            placeholder_id = f"node-{t.provisioned}"
            if placeholder_id in self._instance_costs:
                del self._instance_costs[placeholder_id]

            t.provisioned += 1
            if inst.spot:
                t.spot += 1
            else:
                t.ondemand += 1
            if inst.spec:
                t.instance_types.add(inst.spec.name)

                # Register cost tracking (prices from spec) - billing starts now
                hourly = inst.spec.price_spot if inst.spot else inst.spec.price_on_demand
                self._instance_costs[inst.instance_id] = _InstanceCost(
                    instance_id=inst.instance_id,
                    hourly_rate=hourly or 0.0,
                    on_demand_rate=inst.spec.price_on_demand or 0.0,
                    billing_increment_minutes=inst.spec.billing_increment_minutes,
                )
                self._instance_costs[inst.instance_id].start_time = time.monotonic()

                # Capture infra spec from first instance
                if t.infra_spec is None:
                    t.infra_spec = InfraSpec(
                        provider=inst.provider.value.lower(),
                        region=t.infra_region,
                        instance_type=inst.spec.name,
                        vcpus=inst.spec.vcpu or 0,
                        memory_gb=int(inst.spec.memory_gb or 0),
                        gpu_count=int(inst.spec.accelerator_count or 0),
                        gpu_model=inst.spec.accelerator or "",
                        gpu_vram_gb=int(inst.spec.accelerator_memory_gb or 0),
                        price_per_node=hourly or 0.0,
                    )

            spot_label = "spot" if inst.spot else "od"
            spec_name = inst.spec.name if inst.spec else "?"

            milestone = Text()
            milestone.append("✓ ", style=STYLE_SUCCESS)
            milestone.append(_short_id(inst.instance_id), style="bold")
            milestone.append(f" [{spec_name}, {spot_label}]", style=STYLE_DIM)
            self._add_milestone(milestone)

            t.action = f"Provisioned {t.provisioned}/{t.total_nodes or '?'}"
            self._refresh()

    @handle.register
    def _(self, event: ProvisioningCompleted) -> None:
        with self._lock:
            t = self._tracking
            t.phase = "Bootstrapping"
            t.action = "Starting bootstrap..."
            t.infra_region = event.region

            # Track phase time
            elapsed = time.monotonic() - t.start_time if t.start_time else 0.0
            t.phase_times["provision"] = elapsed

            self._refresh()

    @handle.register
    def _(self, event: BootstrapStarting) -> None:
        with self._lock:
            inst_id = _short_id(event.instance.instance_id)
            self._tracking.action = f"Bootstrapping {inst_id}..."
            self._refresh()

    @handle.register
    def _(self, event: BootstrapProgress) -> None:
        with self._lock:
            t = self._tracking
            t.phase = "Bootstrapping"
            inst_id = _short_id(event.instance.instance_id)

            # Update action with step info
            if event.step == "command" and event.message:
                # For command events, show in logs with $ prefix
                cmd = event.message.strip()
                if cmd:
                    log = Text()
                    log.append(f"[{inst_id}] ", style=STYLE_DIM)
                    log.append("$ ", style="bold cyan")
                    # Truncate long commands
                    display_cmd = cmd[:70] + "..." if len(cmd) > 70 else cmd
                    log.append(display_cmd, style="cyan")
                    self._add_log_line(log)
            elif event.step == "console" and event.message:
                # For console output, show in logs (only non-empty lines)
                msg = event.message.strip()
                if msg and not msg.startswith("#"):
                    log = Text()
                    log.append(f"[{inst_id}] ", style=STYLE_DIM)
                    # Truncate long lines
                    display_msg = msg[:80] + "..." if len(msg) > 80 else msg
                    log.append(display_msg, style=STYLE_DIM)
                    self._add_log_line(log)
            else:
                # For phase steps, update action
                t.action = f"Bootstrapping {inst_id} ({event.step})"

            self._refresh()

    @handle.register
    def _(self, event: BootstrapCompleted) -> None:
        with self._lock:
            t = self._tracking
            t.ready += 1
            total = t.total_nodes or t.provisioned

            inst_id = event.instance.instance_id

            milestone = Text()
            milestone.append("✓ ", style=STYLE_SUCCESS)
            milestone.append(_short_id(inst_id), style="bold")
            if event.duration:
                milestone.append(f" ready ({_format_duration(event.duration)})", style=STYLE_DIM)
            else:
                milestone.append(" ready", style=STYLE_DIM)
            self._add_milestone(milestone)

            if t.ready >= total > 0:
                t.phase = "Executing"
                t.action = "All nodes ready"
            else:
                t.action = f"Bootstrap {t.ready}/{total}"

            self._refresh()

    @handle.register
    def _(self, event: InstanceReady) -> None:
        with self._lock:
            self._refresh()

    @handle.register
    def _(self, event: PoolReady) -> None:
        with self._lock:
            t = self._tracking
            t.phase = "Executing"
            t.ready = len(event.instances)

            # Track bootstrap phase time
            elapsed = time.monotonic() - t.start_time if t.start_time else 0.0
            t.phase_times["bootstrap"] = elapsed

            # Start execute phase timing from now
            t.execute_start_time = time.monotonic()

            duration = _format_duration(event.total_duration_seconds)
            t.action = f"Pool ready in {duration}"

            milestone = Text()
            milestone.append("● ", style=STYLE_INFO)
            milestone.append(f"Ready: {len(event.instances)} nodes", style=STYLE_DIM)
            self._add_milestone(milestone)

            self._refresh()

    @handle.register
    def _(self, event: Metrics) -> None:
        from loguru import logger

        inst_id = event.instance.instance_id
        logger.debug(f"[panel] Metrics received for {inst_id}: {event}%")

        with self._lock:
            t = self._tracking

            # Check if this instance is registered in _instance_costs
            if inst_id not in self._instance_costs:
                logger.warning(
                    f"[panel] Metrics for {inst_id} but not in _instance_costs! "
                    f"Registered: {list(self._instance_costs.keys())}"
                )

            # Calculate VRAM percentage
            vram_pct = 0.0
            if event.gpu_memory_used_mb is not None and event.gpu_memory_total_mb:
                vram_pct = (event.gpu_memory_used_mb / event.gpu_memory_total_mb) * 100

            # GPU temperature (default to 0 if not available)
            gpu_temp = event.gpu_temperature or 0.0

            t.node_metrics[event.instance.node] = _NodeMetrics(
                cpu=event.cpu_percent,
                gpu=event.gpu_utilization,
                mem=event.memory_percent,
                gpu_mem=vram_pct,
                gpu_temp=gpu_temp,
            )

            # Track metrics history for sparklines with EMA smoothing
            raw_cpu = event.cpu_percent or 0.0
            raw_gpu = event.gpu_utilization or 0.0
            raw_vram = vram_pct
            raw_mem = event.memory_percent or 0.0
            raw_temp = gpu_temp

            # Apply EMA smoothing (α=0.15) to reduce spiky transitions
            alpha = 0.15
            if inst_id in t.smoothed_metrics:
                prev_cpu, prev_gpu, prev_vram, prev_mem, prev_temp = t.smoothed_metrics[inst_id]
                cpu = alpha * raw_cpu + (1 - alpha) * prev_cpu
                gpu = alpha * raw_gpu + (1 - alpha) * prev_gpu
                vram = alpha * raw_vram + (1 - alpha) * prev_vram
                mem = alpha * raw_mem + (1 - alpha) * prev_mem
                temp = alpha * raw_temp + (1 - alpha) * prev_temp
            else:
                cpu, gpu, vram, mem, temp = raw_cpu, raw_gpu, raw_vram, raw_mem, raw_temp

            t.smoothed_metrics[inst_id] = (cpu, gpu, vram, mem, temp)

            if inst_id not in t.metrics_history:
                t.metrics_history[inst_id] = []
            t.metrics_history[inst_id].append((cpu, gpu, vram, mem, temp))
            # Keep only last 30 points for sparkline (more history)
            t.metrics_history[inst_id] = t.metrics_history[inst_id][-30:]

            t.action = "Executing..."
            self._refresh()

    @handle.register
    def _(self, event: LogLine) -> None:
        with self._lock:
            if event.line.strip():
                log = Text()
                log.append(f"[{_short_id(event.instance.instance_id)}] ", style=STYLE_INFO)
                log.append(event.line.strip(), style=STYLE_DIM)
                self._add_log_line(log)  # Goes to log_lines, not milestones
                self._refresh()

    @handle.register
    def _(self, event: FunctionCall) -> None:
        with self._lock:
            self._tracking.recent_calls.append(event.function_name)
            # Keep only last 3
            self._tracking.recent_calls = self._tracking.recent_calls[-3:]
            self._refresh()

    @handle.register
    def _(self, event: FunctionResult) -> None:
        with self._lock:
            if not event.success:
                error_text = Text()
                error_text.append("✗ ", style=STYLE_ERROR)
                error_text.append(
                    f"{event.function_name} failed: {event.error or 'unknown error'}",
                    style="red",
                )
                self._add_milestone(error_text)
            self._refresh()

    @handle.register
    def _(self, event: InstanceStopping) -> None:
        with self._lock:
            t = self._tracking
            t.phase = "Shutting down"

            # Stop billing for this instance
            inst_id = event.instance.instance_id
            if inst_id in self._instance_costs:
                self._instance_costs[inst_id].end_time = time.monotonic()

            t.action = f"Stopping {_short_id(inst_id)}..."
            self._refresh()

    @handle.register
    def _(self, event: Error) -> None:
        with self._lock:
            t = self._tracking
            t.has_error = True
            t.phase = "Error"
            t.action = ""

            error_text = Text()
            error_text.append("✗ ", style=STYLE_ERROR)
            error_text.append(event.message[:50], style="red")
            self._add_milestone(error_text)

            self._refresh()


def panel() -> Callback:
    """Create a Rich panel callback for beautiful terminal UI.

    The panel shows a 2-column status display with:
    - Header: spinner emoji + current action + cost/duration
    - Left column: metrics (Nodes, CPU, GPU, Mem, GMem)
    - Right column: milestones (✓ Network ready, ✓ provisioned, etc)
    - Below panel: stdout from nodes (LogLine events)

    Cost tracking is built-in using prices from InstanceSpec events.
    No need to compose with cost_tracker().

    Returns:
        A callback that renders an animated Rich panel.

    Example:
        with use_callback(panel()):
            emit(PoolStarted())
            # ... panel updates ...
            emit(PoolStopping())
    """
    return PanelController().handle
