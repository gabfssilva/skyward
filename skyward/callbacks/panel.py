"""Rich-based panel callback for beautiful terminal UI."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import singledispatchmethod
from math import ceil
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING

from rich.console import Console, Group, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from skyward.callbacks._panel_components import (
    InfraSpec,
    InstanceMetrics,
    LogEntry,
    PhaseInfo,
    create_event_tree,
    create_header,
    create_infra_section,
    create_logs_section,
    create_metrics_table,
    create_panel,
    format_duration,
    render_sparkline,
)
from skyward.events import (
    BootstrapCompleted,
    BootstrapProgress,
    BootstrapStarting,
    Error,
    FunctionCall,
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
    from skyward.callback import Callback

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
    metrics_history: dict[str, list[tuple[float, float, float]]] = field(
        default_factory=dict
    )  # instance_id -> [(cpu, gpu, mem), ...]
    phase_times: dict[str, float] = field(default_factory=dict)  # phase_name -> elapsed
    recent_calls: list[str] = field(default_factory=list)  # last 3 function names
    infra_spec: InfraSpec | None = None
    infra_region: str = ""  # from ProvisioningCompleted
    start_time: float = 0.0  # pool start time for elapsed calculation
    execute_start_time: float = 0.0  # set at PoolReady for execute phase timing


class PanelController:
    """Controls the Rich Live display with 2-column layout."""

    _MAX_LOG_LINES = 50
    _SPINNER_FRAMES = ("🟢", "⚪")

    _PANEL_HEIGHT = 13  # Fixed height for panel section

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

    def _build_phase_info(self) -> list[PhaseInfo]:
        """Build phase info list for event tree."""
        t = self._tracking
        phases = []

        # Network phase
        network_time = t.phase_times.get("network", 0.0)
        network_status: str = "completed" if "network" in t.phase_times else "pending"
        if t.phase == "Provisioning" and "network" not in t.phase_times:
            network_status = "in_progress"
        phases.append(PhaseInfo(
            name="network",
            status=network_status,  # type: ignore
            timestamp=format_duration(network_time) if network_time else "",
            sub_items=("vpc",),
        ))

        # Provision phase
        provision_time = t.phase_times.get("provision", 0.0)
        provision_status: str = "completed" if "provision" in t.phase_times else "pending"
        if t.phase == "Provisioning" and "network" in t.phase_times:
            provision_status = "in_progress"
        sub_items = []
        if t.spot:
            sub_items.append(f"{t.spot} spot")
        if t.ondemand:
            sub_items.append(f"{t.ondemand} on-demand")
        phases.append(PhaseInfo(
            name="provision",
            status=provision_status,  # type: ignore
            timestamp=format_duration(provision_time) if provision_time else "",
            sub_items=tuple(sub_items) if sub_items else ("instances",),
        ))

        # Bootstrap phase
        bootstrap_time = t.phase_times.get("bootstrap", 0.0)
        bootstrap_status: str = "completed" if "bootstrap" in t.phase_times else "pending"
        if t.phase == "Bootstrapping":
            bootstrap_status = "in_progress"
        phases.append(PhaseInfo(
            name="bootstrap",
            status=bootstrap_status,  # type: ignore
            timestamp=format_duration(bootstrap_time) if bootstrap_time else "",
            sub_items=("deps", "healthy"),
        ))

        # Execute phase - timestamp from when pool became ready
        execute_elapsed = time.monotonic() - t.execute_start_time if t.execute_start_time else 0.0
        execute_status: str = "in_progress" if t.phase == "Executing" else "pending"
        if self._is_done:
            execute_status = "completed"
        phases.append(PhaseInfo(
            name="execute",
            status=execute_status,  # type: ignore
            timestamp=format_duration(execute_elapsed) if t.execute_start_time else "",
            sub_items=tuple(t.recent_calls[-3:]) if t.recent_calls else (),
        ))

        return phases

    def _build_instance_metrics(self) -> list[InstanceMetrics]:
        """Build instance metrics list for metrics table."""
        t = self._tracking
        instances = []

        # Iterate over ALL provisioned instances, not just those with metrics
        for inst_id, cost_info in self._instance_costs.items():
            history = t.metrics_history.get(inst_id, [])

            cpu_history = tuple(h[0] for h in history) if history else ()
            gpu_history = tuple(h[1] for h in history) if history else ()
            mem_history = tuple(h[2] for h in history) if history else ()

            cpu_current = history[-1][0] if history else 0.0
            gpu_current = history[-1][1] if history else 0.0
            mem_current = history[-1][2] if history else 0.0

            # Determine market type from cost rates
            market = "on-demand" if cost_info.hourly_rate == cost_info.on_demand_rate else "spot"

            instances.append(InstanceMetrics(
                instance_id=inst_id,
                cpu_history=cpu_history,
                gpu_history=gpu_history,
                mem_history=mem_history,
                cpu_current=cpu_current,
                gpu_current=gpu_current,
                mem_current=mem_current,
                market=market,  # type: ignore
            ))

        return instances

    def _build_panel(self) -> Group:
        """Build the main status panel with 2-column layout using new components."""
        t = self._tracking

        # Calculate current cost and elapsed
        elapsed = time.monotonic() - t.start_time if t.start_time else 0.0
        cost, _, _ = self._calculate_cost()
        blink_on = int(time.monotonic() * 2) % 2 == 0

        # Header
        header = create_header(cost, elapsed, blink_on=blink_on and not self._is_done)

        # Infra section
        if t.infra_spec:
            # Update with current node counts and hourly rate
            total_hourly = sum(
                c.hourly_rate for c in self._instance_costs.values()
            )
            infra = create_infra_section(InfraSpec(
                provider=t.infra_spec.provider,
                region=t.infra_spec.region,
                instance_type=t.infra_spec.instance_type,
                vcpus=t.infra_spec.vcpus,
                memory_gb=t.infra_spec.memory_gb,
                gpu_count=t.infra_spec.gpu_count,
                gpu_model=t.infra_spec.gpu_model,
                total_nodes=t.total_nodes or t.provisioned,
                spot_count=t.spot,
                ondemand_count=t.ondemand,
                hourly_rate=total_hourly,
            ))
        else:
            # Fallback during early provisioning
            infra = Text(f"{t.action or t.phase}", style="dim")

        # Metrics table
        instances = self._build_instance_metrics()
        metrics_table = create_metrics_table(
            instances=instances,
            page=0,
            total_pages=1,
            paused=False,
        )

        # Event tree
        phases = self._build_phase_info()
        event_tree = create_event_tree(
            phases=phases,
            recent_calls=t.recent_calls,
            current_time=format_duration(elapsed),
            blink_on=blink_on and not self._is_done,
        )

        # Assemble panel
        return create_panel(
            header=header,
            infra=infra,
            metrics_table=metrics_table,
            event_tree=event_tree,
            logs=None,  # Logs are added in _build_display
        )

    def _build_display(self) -> Layout:
        """Build complete display using Layout for fixed panel + scrolling logs."""
        if self._layout is None:
            return Layout()  # Should not happen

        # Update panel section
        self._layout["panel"].update(self._build_panel())

        # Update logs section
        if self._tracking.log_lines:
            logs_height = max(1, self._console.size.height - self._PANEL_HEIGHT - 1)
            visible_logs = self._tracking.log_lines[-logs_height:]
            log_content = Group(*visible_logs) if visible_logs else Text("")
            self._layout["logs"].update(log_content)
        else:
            self._layout["logs"].update(Text(""))

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
                    self._live.update(self._build_display())

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

            # Create placeholder entries for nodes (will be updated in InstanceProvisioned)
            for i in range(event.nodes):
                placeholder_id = f"node-{i}"
                self._instance_costs[placeholder_id] = _InstanceCost(
                    instance_id=placeholder_id,
                    hourly_rate=0.0,
                    on_demand_rate=0.0,
                )

            # Create layout with panel on top (fixed height), logs below
            self._layout = Layout()
            self._layout.split_column(
                Layout(name="panel", size=self._PANEL_HEIGHT),
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
        with self._lock:
            t = self._tracking
            gpu_mem_pct = None
            if event.gpu_memory_used_mb is not None and event.gpu_memory_total_mb:
                gpu_mem_pct = (event.gpu_memory_used_mb / event.gpu_memory_total_mb) * 100

            t.node_metrics[event.instance.node] = _NodeMetrics(
                cpu=event.cpu_percent,
                gpu=event.gpu_utilization,
                mem=event.memory_percent,
                gpu_mem=gpu_mem_pct,
            )

            # Track metrics history for sparklines
            inst_id = event.instance.instance_id
            if inst_id not in t.metrics_history:
                t.metrics_history[inst_id] = []
            t.metrics_history[inst_id].append((
                event.cpu_percent or 0.0,
                event.gpu_utilization or 0.0,
                event.memory_percent or 0.0,
            ))
            # Keep only last 12 points for sparkline
            t.metrics_history[inst_id] = t.metrics_history[inst_id][-12:]

            t.action = "Executing..."
            self._refresh()

    @handle.register
    def _(self, event: LogLine) -> None:
        with self._lock:
            if event.line.strip():
                log = Text()
                log.append(f"[node {event.instance.node}] ", style=STYLE_INFO)
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
