"""Reusable panel rendering components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from rich.align import Align
from rich.console import Group
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

# Sparkline characters (9 levels)
_SPARKLINE_CHARS = " ▁▂▃▄▅▆▇█"


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
    mem_history: tuple[float, ...] = ()
    cpu_current: float = 0.0
    gpu_current: float = 0.0
    mem_current: float = 0.0
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
    total_nodes: int = 0
    spot_count: int = 0
    ondemand_count: int = 0
    hourly_rate: float = 0.0


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
    blink_on: bool = True,
) -> Text:
    """Create the header section with cost and elapsed time."""
    header = Text()
    marker = "●" if blink_on else "○"
    header.append(f"{marker}  s k y w a r d", style="bold")
    header.append(f"  ·  ${cost:.2f}  ·  {format_duration(elapsed_seconds)}", style="dim")
    return header


def create_infra_section(spec: InfraSpec, width: int = 45) -> Text:
    """Create the infrastructure section with justified text."""
    lines = [
        f"{spec.provider} › {spec.region} › {spec.instance_type}",
    ]

    # Resource line
    if spec.vcpus or spec.memory_gb or spec.gpu_count:
        parts = []
        if spec.vcpus:
            parts.append(f"{spec.vcpus} vCPU")
        if spec.memory_gb:
            parts.append(f"{spec.memory_gb} GB")
        if spec.gpu_count and spec.gpu_model:
            parts.append(f"{spec.gpu_count}× {spec.gpu_model}")
        lines.append(" · ".join(parts))

    # Nodes line
    if spec.total_nodes:
        node_parts = [f"{spec.total_nodes} nodes"]
        market_parts = []
        if spec.spot_count:
            market_parts.append(f"{spec.spot_count} spot")
        if spec.ondemand_count:
            market_parts.append(f"{spec.ondemand_count} on-demand")
        if market_parts:
            node_parts.append(" + ".join(market_parts))
        if spec.hourly_rate:
            node_parts.append(f"${spec.hourly_rate:.2f}/hr")
        lines.append(" · ".join(node_parts))

    justified = [justify_text(line, width) for line in lines]
    return Text("\n".join(justified))


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


def create_metrics_table(
    instances: list[InstanceMetrics],
    page: int = 0,
    total_pages: int = 1,
    paused: bool = False,
) -> Table:
    """Create the metrics table with sparklines."""
    table = Table(
        box=None,
        show_header=True,
        header_style="dim",
        padding=(0, 1),
        expand=False,
    )

    table.add_column("id", width=20)
    table.add_column("cpu", justify="center", width=16)
    table.add_column("gpu", justify="center", width=16)
    table.add_column("mem", justify="center", width=16)
    table.add_column("market", justify="center", width=16)

    for inst in instances:
        cpu_spark = render_sparkline(inst.cpu_history)
        gpu_spark = render_sparkline(inst.gpu_history)
        mem_spark = render_sparkline(inst.mem_history)

        table.add_row(
            Text(inst.instance_id[:16], style="cyan"),
            Text(f"{cpu_spark} {inst.cpu_current:.0f}%", style="green"),
            Text(f"{gpu_spark} {inst.gpu_current:.0f}%", style="magenta"),
            Text(f"{mem_spark} {inst.mem_current:.0f}%", style="blue"),
            Text(inst.market, style="dim"),
        )

    # Pagination indicator
    play_pause = "⏸" if paused else "▶"
    dots = "".join("●" if i == page else "○" for i in range(total_pages))
    page_indicator = f"{play_pause} {dots}"

    # Only show separator + avg row when multiple instances
    if len(instances) > 1:
        # Calculate averages
        avg_cpu = sum(i.cpu_current for i in instances) / len(instances)
        avg_gpu = sum(i.gpu_current for i in instances) / len(instances)
        avg_mem = sum(i.mem_current for i in instances) / len(instances)

        # Calculate average sparklines from histories
        max_len = max((len(i.cpu_history) for i in instances), default=0)
        avg_cpu_history: list[float] = []
        avg_gpu_history: list[float] = []
        avg_mem_history: list[float] = []

        for idx in range(max_len):
            cpu_vals = [i.cpu_history[idx] for i in instances if idx < len(i.cpu_history)]
            gpu_vals = [i.gpu_history[idx] for i in instances if idx < len(i.gpu_history)]
            mem_vals = [i.mem_history[idx] for i in instances if idx < len(i.mem_history)]

            avg_cpu_history.append(sum(cpu_vals) / len(cpu_vals) if cpu_vals else 0.0)
            avg_gpu_history.append(sum(gpu_vals) / len(gpu_vals) if gpu_vals else 0.0)
            avg_mem_history.append(sum(mem_vals) / len(mem_vals) if mem_vals else 0.0)

        table.add_row(
            Text("avg", style="bold"),
            Text(f"{render_sparkline(avg_cpu_history)} {avg_cpu:.0f}%", style="green bold"),
            Text(f"{render_sparkline(avg_gpu_history)} {avg_gpu:.0f}%", style="magenta bold"),
            Text(f"{render_sparkline(avg_mem_history)} {avg_mem:.0f}%", style="blue bold"),
            Text(page_indicator, style="dim"),
        )
    else:
        # Single instance - just show pagination in the last column of that row
        # (pagination already shown per-instance, but we need it somewhere)
        pass

    return table


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
    infra: Text,
    metrics_table: Table,
    event_tree: Tree,
    logs: list[LogEntry] | None = None,
    left_width: int = 60,
    right_width: int = 26,
    padding: int = 4,
) -> Group:
    """Assemble the complete panel layout."""
    separator = Text("─" * 45, style="dim")

    # Left side: header, infra, separator, metrics
    left_content = Group(
        Align.center(header),
        Align.center(infra),
        Align.center(separator),
        metrics_table,
    )

    # Two-column layout
    main_table = Table(box=None, show_header=False, padding=(0, padding), expand=False)
    main_table.add_column("left", width=left_width)
    main_table.add_column("right", width=right_width)
    main_table.add_row(left_content, event_tree)

    elements: list = [Align.center(main_table)]

    if logs:
        elements.append(create_logs_section(logs))

    return Group(*elements)
