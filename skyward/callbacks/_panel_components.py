"""Reusable panel rendering components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rich.align import Align
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

# Sparkline characters (9 levels)
_SPARKLINE_CHARS = "⎽▁▂▃▄▅▆▇█"

# Braille spinner (10 frames)
_SPINNER_CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


@dataclass(frozen=True, slots=True)
class PhaseInfo:
    """Information about a pipeline phase."""

    name: str  # "network", "provision", "bootstrap", "execute"
    status: Literal["pending", "in_progress", "completed"]
    timestamp: str  # "0:12" or ""
    sub_items: tuple[str, ...] = ()  # ("vpc",), ("10 spot", "2 od"), etc.


@dataclass(frozen=True, slots=True)
class InstanceMetrics:
    """Metrics for a single instance with history for sparklines."""

    instance_id: str
    cpu_history: tuple[float, ...] = ()
    gpu_history: tuple[float, ...] = ()
    vram_history: tuple[float, ...] = ()
    mem_history: tuple[float, ...] = ()
    temp_history: tuple[float, ...] = ()
    cpu_current: float = 0.0
    gpu_current: float = 0.0
    vram_current: float = 0.0
    mem_current: float = 0.0
    temp_current: float = 0.0
    market: Literal["spot", "on-demand"] = "spot"


@dataclass(frozen=True, slots=True)
class LogEntry:
    """A log line entry."""

    timestamp: str
    node_id: str
    message: str


@dataclass(frozen=True, slots=True)
class InfraSpec:
    """Infrastructure specification."""

    provider: str  # "aws", "digitalocean", "verda"
    region: str  # "us-east-1"
    instance_type: str  # "p4d.24xlarge"
    vcpus: int = 0
    memory_gb: int = 0
    gpu_count: int = 0
    gpu_model: str = ""
    gpu_vram_gb: int = 0  # VRAM per GPU in GB
    total_nodes: int = 0
    spot_count: int = 0
    ondemand_count: int = 0
    hourly_rate: float = 0.0  # Total hourly rate (all nodes)
    price_per_node: float = 0.0  # Hourly rate per node


def render_sparkline(history: tuple[float, ...] | list[float], width: int = 10) -> str:
    """Convert a history of percentage values (0-100) to a sparkline string."""
    if not history:
        return _SPARKLINE_CHARS[0] * width  # ▁ (baseline) instead of ░

    # Take last `width` values
    values = list(history)[-width:]

    # Pad with zeros if not enough values
    while len(values) < width:
        values.insert(0, 0.0)

    # Convert each value to a sparkline character
    chars = []
    for v in values:
        # Clamp to 0-100
        v = max(0.0, min(100.0, v))
        # Map to 0-7 index
        idx = int((v / 100.0) * 7)
        chars.append(_SPARKLINE_CHARS[idx])

    return "".join(chars)


def justify_text(text: str, width: int) -> str:
    """Spread text horizontally by adding spaces between words."""
    words = text.split()
    if len(words) <= 1:
        return text

    total_chars = sum(len(w) for w in words)
    total_spaces = width - total_chars
    gaps = len(words) - 1

    if gaps <= 0 or total_spaces <= gaps:
        return text

    space_per_gap = total_spaces // gaps
    extra = total_spaces % gaps

    result = []
    for i, word in enumerate(words):
        result.append(word)
        if i < gaps:
            spaces = space_per_gap + (1 if i < extra else 0)
            result.append(" " * spaces)

    return "".join(result)


def format_duration(seconds: float) -> str:
    """Format seconds as M:SS."""
    mins, secs = divmod(int(seconds), 60)
    return f"{mins}:{secs:02d}"


def create_header(
    cost: float,
    elapsed_seconds: float,
    phases: dict[str, Literal["pending", "in_progress", "completed"]],
    tick: int = 0,
    blink_on: bool = True,
) -> Text:
    """Create the header with cost, elapsed time, and current phase status.

    Format: ● skyward · 2:40 · ~ $0.01 · Preparing ⠇
    """
    header = Text()
    marker = "●" if blink_on else "○"
    header.append(f"{marker} ", style="cyan bold")
    header.append("skyward", style="bold")
    header.append(f" · {format_duration(elapsed_seconds)} · ~ ${cost:.2f} · ", style="dim")

    # Find current active phase
    active_phase = None
    for name, status in phases.items():
        if status == "in_progress":
            active_phase = name
            break

    # Status and spinner
    if active_phase is None:
        all_completed = all(s == "completed" for s in phases.values())
        if all_completed:
            header.append("done", style="green bold")
        else:
            header.append("waiting...", style="dim")
    else:
        # Convert phase name to gerund (Provision -> Provisioning, Execute -> Executing)
        gerund = f"{active_phase[:-1]}ing" if active_phase.endswith("e") else f"{active_phase}ing"
        spinner = _SPINNER_CHARS[tick % len(_SPINNER_CHARS)]
        header.append(gerund.lower(), style="bold")
        header.append(f" {spinner}", style="cyan bold")

    return header


def create_infra_section(spec: InfraSpec) -> tuple[Text, Text, Text, Text, Text]:
    """Create the infrastructure section as 5 separate lines.

    Line 1: provider › region
    Line 2: instance · GPU (VRAM)
    Line 3: allocation @ price/node = total
    Line 4: mini separator
    Line 5: cluster totals (vCPU, RAM, VRAM)
    """
    # Line 1: provider › region
    line1 = Text()
    line1.append(spec.provider, style="yellow")
    line1.append(" › ", style="dim")
    line1.append(spec.region)

    # Line 2: instance · GPU (VRAM)
    line2 = Text()
    line2.append(spec.instance_type, style="bold")
    if spec.gpu_count and spec.gpu_model:
        line2.append(" · ", style="dim")
        line2.append(f"{spec.gpu_count}× {spec.gpu_model}", style="magenta")
        if spec.gpu_vram_gb:
            vram_per_node = spec.gpu_count * spec.gpu_vram_gb
            line2.append(f" ({vram_per_node} GB)", style="dim")

    # Line 3: allocation @ price/node = total
    line3 = Text()
    if spec.total_nodes:
        if spec.spot_count > 0 and spec.ondemand_count > 0:
            line3.append(f"{spec.spot_count} spot", style="cyan")
            line3.append(" + ", style="dim")
            line3.append(f"{spec.ondemand_count} od", style="yellow")
        elif spec.spot_count > 0:
            line3.append(f"{spec.spot_count} spot", style="cyan")
        else:
            line3.append(f"{spec.ondemand_count} on-demand", style="yellow")

        if spec.price_per_node and spec.hourly_rate:
            line3.append(f" @ ${spec.price_per_node:.2f}/hr", style="dim")
            line3.append(" = ", style="dim")
            if spec.hourly_rate < 10:
                line3.append(f"${spec.hourly_rate:.2f}/hr", style="green bold")
            else:
                line3.append(f"${spec.hourly_rate:.0f}/hr", style="green bold")

    # Line 4: mini separator
    line4 = Text("─" * 28, style="dim")

    # Line 5: cluster totals
    line5 = Text()
    n = spec.total_nodes or 1
    parts: list[str] = []
    if spec.vcpus:
        parts.append(f"{spec.vcpus * n} vCPU")
    if spec.memory_gb:
        parts.append(f"{spec.memory_gb * n} GB RAM")
    if spec.gpu_count and spec.gpu_vram_gb:
        total_vram = spec.gpu_count * spec.gpu_vram_gb * n
        parts.append(f"{total_vram} GB VRAM")
    line5.append(" · ".join(parts), style="bold")

    return line1, line2, line3, line4, line5


def create_event_tree(
    phases: list[PhaseInfo],
    recent_calls: list[str],
    current_time: str,
    blink_on: bool = True,
) -> Tree:
    """Create the event tree using Rich Tree (max 12 lines)."""
    tree = Tree("", guide_style="dim", hide_root=True)

    for phase in phases:
        # Determine marker based on status
        if phase.status == "completed":
            marker = "[green]✓[/]"
            style = "dim"
        elif phase.status == "in_progress":
            dot = "◉" if blink_on else "○"
            marker = f"[cyan bold]{dot}[/]"
            style = "bold"
        else:
            marker = "[dim]○[/]"
            style = "dim"

        # Phase label with timestamp
        timestamp = f" {phase.timestamp}" if phase.timestamp else ""
        label = f"{marker} [{style}]{phase.name}{timestamp}[/]"
        branch = tree.add(label)

        # Add sub-items
        if phase.status == "in_progress" and phase.name == "execute":
            # For execute phase, show recent function calls
            for fn in recent_calls[-3:]:
                branch.add(f"[white]{fn[:12]}()[/]")
        else:
            for item in phase.sub_items:
                branch.add(f"[dim]{item}[/]")

    return tree


def _temp_color(temp: float) -> str:
    """Return color based on temperature: green < 60, yellow 60-75, red > 75."""
    if temp < 60:
        return "green"
    if temp < 75:
        return "yellow"
    return "red"


def _make_metrics_cells(
    inst: InstanceMetrics,
    spark_width: int,
) -> tuple[Text, Text, Text, Text, Text, Text]:
    """Create cell contents for a single instance row (6 cells for Table columns)."""
    # Column 1: NODE
    node = Text()
    id_style = "cyan" if inst.market == "spot" else "yellow"
    node.append(inst.instance_id[:10], style=id_style)

    # Column 2: CPU
    cpu = Text()
    cpu.append(render_sparkline(inst.cpu_history, spark_width), style="green")
    cpu.append(f" {inst.cpu_current:>3.0f}%", style="green bold")

    # Column 3: GPU
    gpu = Text()
    gpu.append(render_sparkline(inst.gpu_history, spark_width), style="magenta")
    gpu.append(f" {inst.gpu_current:>3.0f}%", style="magenta bold")

    # Column 4: VRAM
    vram = Text()
    vram.append(render_sparkline(inst.vram_history, spark_width), style="blue")
    vram.append(f" {inst.vram_current:>3.0f}%", style="blue bold")

    # Column 5: MEM
    mem = Text()
    mem.append(render_sparkline(inst.mem_history, spark_width), style="cyan")
    mem.append(f" {inst.mem_current:>3.0f}%", style="cyan bold")

    # Column 6: TEMP (simple value, no sparkline)
    temp = Text()
    temp.append(f"{inst.temp_current:>3.0f}°", style=f"{_temp_color(inst.temp_current)} bold")

    return node, cpu, gpu, vram, mem, temp


def _make_cluster_cells(
    instances: list[InstanceMetrics],
    spark_width: int,
) -> tuple[Text, Text, Text, Text, Text, Text]:
    """Create cell contents for the cluster average row (6 cells for Table columns).

    Args:
        instances: All instances to compute average from.
        spark_width: Width of sparkline charts.
    """
    if not instances:
        return Text(), Text(), Text(), Text(), Text(), Text()

    # Calculate averages
    avg_cpu = sum(i.cpu_current for i in instances) / len(instances)
    avg_gpu = sum(i.gpu_current for i in instances) / len(instances)
    avg_vram = sum(i.vram_current for i in instances) / len(instances)
    avg_mem = sum(i.mem_current for i in instances) / len(instances)
    avg_temp = sum(i.temp_current for i in instances) / len(instances)

    # Calculate average sparklines
    max_len = max((len(i.cpu_history) for i in instances), default=0)
    avg_cpu_history: list[float] = []
    avg_gpu_history: list[float] = []
    avg_vram_history: list[float] = []
    avg_mem_history: list[float] = []

    for idx in range(max_len):
        cpu_vals = [i.cpu_history[idx] for i in instances if idx < len(i.cpu_history)]
        gpu_vals = [i.gpu_history[idx] for i in instances if idx < len(i.gpu_history)]
        vram_vals = [i.vram_history[idx] for i in instances if idx < len(i.vram_history)]
        mem_vals = [i.mem_history[idx] for i in instances if idx < len(i.mem_history)]

        avg_cpu_history.append(sum(cpu_vals) / len(cpu_vals) if cpu_vals else 0.0)
        avg_gpu_history.append(sum(gpu_vals) / len(gpu_vals) if gpu_vals else 0.0)
        avg_vram_history.append(sum(vram_vals) / len(vram_vals) if vram_vals else 0.0)
        avg_mem_history.append(sum(mem_vals) / len(mem_vals) if mem_vals else 0.0)

    # Column 1: NODE (average label)
    node = Text()
    node.append("avg", style="dim")

    # Column 2: CPU
    cpu = Text()
    cpu.append(render_sparkline(avg_cpu_history, spark_width), style="green")
    cpu.append(f" {avg_cpu:>3.0f}%", style="green bold")

    # Column 3: GPU
    gpu = Text()
    gpu.append(render_sparkline(avg_gpu_history, spark_width), style="magenta")
    gpu.append(f" {avg_gpu:>3.0f}%", style="magenta bold")

    # Column 4: VRAM
    vram = Text()
    vram.append(render_sparkline(avg_vram_history, spark_width), style="blue")
    vram.append(f" {avg_vram:>3.0f}%", style="blue bold")

    # Column 5: MEM
    mem = Text()
    mem.append(render_sparkline(avg_mem_history, spark_width), style="cyan")
    mem.append(f" {avg_mem:>3.0f}%", style="cyan bold")

    # Column 6: TEMP (simple value, no sparkline)
    temp = Text()
    temp.append(f"{avg_temp:>3.0f}°", style=f"{_temp_color(avg_temp)} bold")

    return node, cpu, gpu, vram, mem, temp


def _make_pagination_row(
    visible_count: int,
    total_count: int,
    current_page: int,
    total_pages: int,
) -> tuple[Text, Text, Text, Text, Text, Text]:
    """Create pagination separator row spanning all columns."""
    # First column: node count

    total_nodes = Text(f"{visible_count}/{total_count} nodes")
    total_pages = Text(f"Page {current_page}/{total_pages}")

    return Text("─" * 3), Text(), Text(), total_nodes, total_pages, Text()

def create_metrics_table(
    visible_instances: list[InstanceMetrics],
    all_instances: list[InstanceMetrics],
    spark_width: int,
    max_visible: int,
    page_info: tuple[int, int] | None = None,
) -> Table:
    """Create the metrics table with Rich Table.

    Args:
        visible_instances: Current page of instances to display.
        all_instances: All instances (used for cluster average calculation).
        spark_width: Width of sparkline charts.
        max_visible: Maximum visible instances per page (for padding).
        page_info: Optional (current_page, total_pages) tuple for pagination indicator.
    """
    col_width = spark_width + 5  # spark + space + value
    temp_width = 4  # just "XX°" value, no sparkline

    table = Table(
        box=None,
        show_header=True,
        show_edge=False,
        padding=(0, 1),
        expand=False,
    )

    # Sparkline columns have equal width, TEMP is smaller (no sparkline)
    table.add_column("#", style="dim", no_wrap=True, width=col_width, justify="center")
    table.add_column("cpu", style="bold", no_wrap=True, width=col_width, justify="center")
    table.add_column("gpu", style="bold", no_wrap=True, width=col_width, justify="center")
    table.add_column("vram", style="bold", no_wrap=True, width=col_width, justify="center")
    table.add_column("ram", style="bold", no_wrap=True, width=col_width, justify="center")
    table.add_column("temp", style="dim", no_wrap=True, width=temp_width, justify="center")

    # Instance rows (current page only)
    for inst in visible_instances:
        table.add_row(*_make_metrics_cells(inst, spark_width))

    # Pad with empty rows to keep average at fixed position
    padding_needed = max_visible - len(visible_instances)
    for _ in range(padding_needed):
        table.add_row()

    # Cluster average row (computed from ALL instances, not just visible)
    # Always shown when there are multiple total instances
    if len(all_instances) > 1:
        # Separator with pagination info spanning all columns
        if page_info:
            current_page, total_pages = page_info
            pagination_row = _make_pagination_row(
                visible_count=len(visible_instances),
                total_count=len(all_instances),
                current_page=current_page,
                total_pages=total_pages,
            )
            table.add_row(*pagination_row)
        else:
            table.add_row()  # Empty row for visual separation

        avg_row = _make_cluster_cells(all_instances, spark_width)
        table.add_row(*avg_row)

    return table


def create_left_table(
    header: Text,
    infra: tuple[Text, Text, Text, Text, Text],
    left_col: int = 46,
) -> Table:
    """Create the left column content as a Table for alignment (7 rows).

    Header is left-aligned, infra lines are centered.
    """
    infra1, infra2, infra3, infra4, infra5 = infra

    table = Table(box=None, show_header=False, padding=0, expand=False)
    table.add_column(width=left_col, no_wrap=True)

    table.add_row(header)
    table.add_row(Text("─" * left_col, style="dim"))
    table.add_row(Align.center(infra1))
    table.add_row(Align.center(infra2))
    table.add_row(Align.center(infra3))
    table.add_row(Align.center(infra4))
    table.add_row(Align.center(infra5))

    return table


def calculate_spark_width(terminal_width: int, left_col: int = 46) -> int:
    """Calculate sparkline width based on terminal size for box.MINIMAL Table."""
    spark_columns = 5  # NODE, CPU, GPU, VRAM, MEM (with sparklines)
    value_width = 5  # " XX%" (space + 3 digits + symbol)
    temp_width = 4  # "XX°" (no sparkline)
    outer_padding = 4  # padding between left and right columns (1 char × 2 sides × 2 cols)
    cell_padding = 6 * 2  # padding=(0,1) adds 1 char each side × 6 cols

    fixed = (
        left_col + outer_padding
        + spark_columns * value_width
        + temp_width
        + cell_padding
    )
    available = terminal_width - fixed
    spark_width = available // spark_columns
    return max(6, spark_width)


def create_logs_section(logs: list[LogEntry]) -> Text:
    """Create the logs section."""
    text = Text()
    for log in logs:
        text.append(f" {log.timestamp} ", style="dim")
        text.append(f"{log.node_id} ", style="cyan")
        text.append(f"{log.message}\n", style="white")
    return text


def create_panel(
    *,
    header: Text,
    infra: tuple[Text, Text, Text, Text, Text],
    visible_instances: list[InstanceMetrics],
    all_instances: list[InstanceMetrics],
    max_visible: int,
    page_info: tuple[int, int] | None = None,
    terminal_width: int = 120,
    left_col: int = 46,
) -> Table:
    """Assemble the complete panel layout with nested Tables.

    Args:
        header: Header text with status info.
        infra: Infrastructure lines (5 lines).
        visible_instances: Current page of instances to display.
        all_instances: All instances (for cluster average).
        max_visible: Maximum visible instances per page (for padding).
        page_info: Optional (current_page, total_pages) for pagination.
        terminal_width: Terminal width for sizing.
        left_col: Width of left column.
    """
    spark_width = calculate_spark_width(terminal_width, left_col)

    # Create left column table (7 rows: header, separator, infra×5)
    left = create_left_table(header, infra, left_col)

    # Create right column metrics table with pagination
    right = create_metrics_table(
        visible_instances, all_instances, spark_width, max_visible, page_info
    )

    # Combine in outer table with separation between left and right
    outer = Table(box=None, show_header=False, padding=(0, 1), expand=False)
    outer.add_column("left", width=left_col, no_wrap=True, vertical="top")
    outer.add_column("right", no_wrap=True, vertical="top")
    outer.add_row(left, right)

    return outer
