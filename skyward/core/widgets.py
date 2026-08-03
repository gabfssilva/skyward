"""The console's Rich renderables — logo, badges, node tables, the bootstrap timeline.

Pure drawing: nothing here knows about the daemon or its events; ``live`` owns the
state and hands it in.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto
from types import MappingProxyType

from rich.align import Align
from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.style import Style
from rich.table import Table
from rich.text import Text


class _Phase(Enum):
    PROVISIONING = auto()
    SSH = auto()
    BOOTSTRAP = auto()
    WORKERS = auto()
    READY = auto()
    STOPPED = auto()


class _NodeStatus(Enum):
    WAITING = auto()
    SSH = auto()
    BOOTSTRAPPING = auto()
    READY = auto()


@dataclass(frozen=True, slots=True)
class _Accelerator:
    name: str
    count: float = 1.0
    memory: str = ""


@dataclass(frozen=True, slots=True)
class _InstanceType:
    name: str
    vcpus: float
    memory_gb: float
    accelerator: _Accelerator | None = None


@dataclass(frozen=True, slots=True)
class _Offer:
    instance_type: _InstanceType
    spot_price: float | None = None
    on_demand_price: float | None = None


@dataclass(frozen=True, slots=True)
class _Instance:
    id: str
    ip: str | None
    ssh_port: int
    region: str
    spot: bool
    offer: _Offer


@dataclass(frozen=True, slots=True)
class _ClusterSpec:
    provider: str


@dataclass(frozen=True, slots=True)
class _Cluster:
    spec: _ClusterSpec


@dataclass(frozen=True, slots=True)
class _BootstrapTimeline:
    phases: tuple[str, ...]
    completed: frozenset[str]
    active: str
    output: str


@dataclass(frozen=True, slots=True)
class _State:
    total_nodes: int
    phase: _Phase = _Phase.PROVISIONING
    nodes: MappingProxyType[int, _NodeStatus] = MappingProxyType({})
    tasks_queued: int = 0
    tasks_running: int = 0
    tasks_done: int = 0
    tasks_failed: int = 0
    first_task_at: float = 0.0
    cluster: _Cluster | None = None
    instances: tuple[_Instance, ...] = ()
    metrics: MappingProxyType[int, MappingProxyType[str, float]] = MappingProxyType({})
    pool_started_at: float = 0.0
    task_latencies: tuple[float, ...] = ()
    task_fn_stats: MappingProxyType[str, tuple[float, ...]] = MappingProxyType({})
    task_fn_failed: MappingProxyType[str, int] = MappingProxyType({})
    ready_at: float = 0.0
    desired_nodes: int = 0
    pending_nodes: int = 0
    draining_nodes: int = 0
    reconciler_state: str = "watching"
    min_nodes: int | None = None
    max_nodes: int | None = None
    is_elastic: bool = False
    spec_accelerator_memory: str = ""
    tasks_per_node: MappingProxyType[int, int] = MappingProxyType({})
    ssh_user: str = ""
    ssh_key_path: str = ""
    bootstrap_spinners: MappingProxyType[int, _BootstrapTimeline] = MappingProxyType({})
    progress_lines: MappingProxyType[int, str] = MappingProxyType({})
    node_instances: MappingProxyType[int, _Instance] = MappingProxyType({})


def _throughput(state: _State, now: float | None = None) -> float:
    if not state.tasks_done:
        return 0.0
    ts = now if now is not None else time.monotonic()
    elapsed_min = (ts - state.first_task_at) / 60
    return state.tasks_done / elapsed_min if elapsed_min > 0 else 0.0


_LOGO_LINES = (
    "   ▌           ▌",
    " ▛▘▙▘▌▌▌▌▌▀▌▛▘▛▌",
    " ▄▌▛▖▙▌▚▚▘█▌▌ ▙▌",
    "     ▄▌",
)

DIM = Style(color="bright_black")
MEDIUM = Style(color="white")
WARNING_STYLE = "yellow"


def _detect_terminal_bg() -> tuple[int, int, int] | None:
    import os
    import select
    import sys
    import termios
    import tty

    try:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
    except (OSError, termios.error):
        return None

    try:
        tty.setraw(fd)
        os.write(sys.stdout.fileno(), b"\033]11;?\033\\")
        if not select.select([fd], [], [], 0.5)[0]:
            return None
        response = b""
        while select.select([fd], [], [], 0.1)[0]:
            response += os.read(fd, 1024)
        decoded = response.decode("latin-1")
        if "rgb:" not in decoded:
            return None
        parts = decoded.split("rgb:")[1].split("\033")[0].split("\\")[0].split("/")
        if len(parts) != 3:
            return None
        return int(parts[0][:2], 16), int(parts[1][:2], 16), int(parts[2][:2], 16)
    except Exception:
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _is_dark_background() -> bool:
    background = _detect_terminal_bg()
    if background is None:
        return True
    red, green, blue = background
    return 0.299 * red + 0.587 * green + 0.114 * blue < 128


_DARK = _is_dark_background()
_BADGE_L = 0.55 if _DARK else 0.35
_BADGE_FG = "rgb(0,0,0)" if _DARK else "rgb(255,255,255)"
WARNING_STYLE = "yellow" if _DARK else "dark_orange"

_FIXED_BADGE_HUES: dict[str, tuple[float, float]] = {
    "skyward": (255.0, 0.50),
    "cluster": (150.0, 0.45),
    "error": (0.0, 0.60),
    "local": (0.0, 0.0),
    "queued": (30.0, 0.50),
    "running": (210.0, 0.60),
    "done": (120.0, 0.50),
    "failed": (0.0, 0.60),
    "connecting": (45.0, 0.45),
    "bootstrap": (45.0, 0.45),
    "provisioning": (45.0, 0.45),
    "ready": (120.0, 0.50),
    "scaling": (45.0, 0.45),
    "in sync": (120.0, 0.50),
    "drifted": (0.0, 0.60),
    "shutting down": (0.0, 0.0),
    "cost": (0.0, 0.0),
}


def _hsl_to_rgb(hue: float, saturation: float, lightness: float) -> tuple[int, int, int]:
    chroma = (1 - abs(2 * lightness - 1)) * saturation
    secondary = chroma * (1 - abs((hue / 60) % 2 - 1))
    match hue:
        case value if value < 60:
            red, green, blue = chroma, secondary, 0.0
        case value if value < 120:
            red, green, blue = secondary, chroma, 0.0
        case value if value < 180:
            red, green, blue = 0.0, chroma, secondary
        case value if value < 240:
            red, green, blue = 0.0, secondary, chroma
        case value if value < 300:
            red, green, blue = secondary, 0.0, chroma
        case _:
            red, green, blue = chroma, 0.0, secondary
    offset = lightness - chroma / 2
    return int((red + offset) * 255), int((green + offset) * 255), int((blue + offset) * 255)


def _make_badge(hue: float, saturation: float) -> Style:
    red, green, blue = _hsl_to_rgb(hue % 360, saturation, _BADGE_L)
    return Style(color=_BADGE_FG, bgcolor=f"rgb({red},{green},{blue})", bold=True)


_FIXED_BADGES = {name: _make_badge(hue, saturation) for name, (hue, saturation) in _FIXED_BADGE_HUES.items()}
_FIXED_BADGES["skyward"] = Style(
    color="rgb(0,0,0)" if _DARK else "rgb(255,255,255)",
    bgcolor="rgb(255,255,255)" if _DARK else "rgb(0,0,0)",
    bold=True,
)


def _stable_hash(label: str) -> int:
    import hashlib

    return int.from_bytes(hashlib.md5(label.encode()).digest()[:4], "big")


def _badge_style(label: str) -> Style:
    if label in _FIXED_BADGES:
        return _FIXED_BADGES[label]
    if label.startswith("~$"):
        return _FIXED_BADGES["cost"]
    return _make_badge((_stable_hash(label) * 137.508) % 360, 0.65)


def _gauge_badge(_label: str, percentage: float) -> Style:
    return _make_badge(120.0 * (1 - min(percentage, 100.0) / 100.0), 0.55)


_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_spinner_tick = [0]
_PHASE_LABELS = {
    "connecting": "ssh",
    "env": "set env",
    "apt": "apt install",
    "uv": "setup uv",
    "venv": "create venv",
    "deps": "install deps",
    "skyward": "install skyward",
    "volumes": "mount volumes",
    "worker": "start worker",
}


def _badge_text(label: str, link: str = "") -> Text:
    short = label[:8].center(8) if len(label) > 8 else label.center(8)
    style = _badge_style(label)
    if link:
        style += Style(link=link)
    text = Text()
    text.append(f" {short} ", style=style)
    return text


def _node_instance(state: _State, node_id: int) -> _Instance | None:
    if node_id in state.node_instances:
        return state.node_instances[node_id]
    if 0 <= node_id < len(state.instances):
        return state.instances[node_id]
    return None


def _node_label(state: _State, node_id: int) -> str:
    instance = _node_instance(state, node_id)
    return instance.id if instance else f"node-{node_id}"


def _ssh_url(state: _State, node_id: int) -> str:
    instance = _node_instance(state, node_id)
    if not instance or not instance.ip or not state.ssh_user:
        return ""
    port = f":{instance.ssh_port}" if instance.ssh_port != 22 else ""
    return f"ssh://{state.ssh_user}@{instance.ip}{port}"


def _inline_badge(label: str) -> Text:
    text = Text()
    text.append(f" {label} ", style=_badge_style(label))
    return text


def _emit(console: Console, badge: str, text: str, style: str = "", link: str = "") -> None:
    line = _badge_text(badge, link=link)
    text_style = Style.parse(style) + Style(link=link) if link else (style or None)
    line.append(f"  {text}", style=text_style)
    console.print(line)


def _emit_task(
    console: Console,
    badge: str,
    status: str,
    text: str,
    link: str = "",
    cost: str = "",
) -> None:
    line = _badge_text(badge, link=link)
    line.append(" ")
    line.append_text(_inline_badge(status))
    if cost:
        line.append(" ")
        line.append_text(_inline_badge(cost))
    line.append(f" {text}", style=Style(link=link) if link else None)
    console.print(line)


def _find_metrics(raw: MappingProxyType[str, float], *prefixes: str) -> list[float]:
    values: list[float] = []
    for prefix in prefixes:
        if prefix in raw:
            values.append(raw[prefix])
        else:
            values.extend(value for key, value in raw.items() if key.startswith(f"{prefix}_"))
    return values


def _collect_metric_vals(state: _State, *prefixes: str) -> list[float]:
    values: list[float] = []
    for metrics in state.metrics.values():
        values.extend(_find_metrics(metrics, *prefixes))
    return values


def _cost_badges(state: _State) -> tuple[Text, Text] | None:
    if not state.instances:
        return None
    hourly = sum(
        (instance.offer.spot_price if instance.spot else instance.offer.on_demand_price) or 0.0
        for instance in state.instances
    )
    if hourly <= 0:
        return None
    elapsed_hours = (time.monotonic() - state.pool_started_at) / 3600 if state.pool_started_at else 0
    return (
        Text(f" ${hourly:.2f}/hr ", style=_badge_style("cost")),
        Text(f" Σ ${hourly * elapsed_hours:.2f} ", style=_badge_style("cost")),
    )


def _gauge_inline(label: str, percentage: float) -> Text:
    text = Text()
    text.append(f" {label} {percentage:.0f}% ", style=_gauge_badge(label, percentage))
    return text


def _styled_badge(label: str, style_key: str) -> Text:
    text = Text()
    text.append(f" {label} ", style=_badge_style(style_key))
    return text


def _progress_badge(label: str, current: int, total: int) -> Text:
    if total > 0 and current >= total:
        style = _make_badge(120, 0.50)
    elif current == 0:
        style = _make_badge(0, 0.60)
    else:
        style = _make_badge(45, 0.45)
    text = Text()
    text.append(f" {label} {current}/{total} ", style=style)
    return text


def _collect_badges(state: _State) -> tuple[list[Text], list[Text], list[Text]]:
    infra: list[Text] = []
    status: list[Text] = []
    tasks: list[Text] = []

    if state.phase in {_Phase.READY, _Phase.STOPPED}:
        infra.append(_inline_badge("skyward"))
    else:
        frame = _SPINNER_FRAMES[_spinner_tick[0] % len(_SPINNER_FRAMES)]
        infra.append(Text(f" {frame} skyward ", style=_badge_style("skyward")))

    if state.instances:
        first = state.instances[0]
        instance_type = first.offer.instance_type
        count = state.desired_nodes or state.total_nodes or len(state.instances)
        spot = sum(1 for instance in state.instances if instance.spot)
        on_demand = count - spot

        infra.append(_styled_badge(f"{count}×", "cluster"))
        if spot:
            infra.append(_inline_badge("spot"))
        if on_demand:
            infra.append(_inline_badge("on-demand"))
        if instance_type.name:
            infra.append(_inline_badge(instance_type.name))
        if first.region:
            infra.append(_inline_badge(first.region))
        if state.cluster and state.cluster.spec.provider:
            infra.append(_inline_badge(f"☁️ {state.cluster.spec.provider.upper()}"))
        if vcpus := int(instance_type.vcpus * count):
            infra.append(_inline_badge(f"{vcpus} vCPU"))
        if memory := int(instance_type.memory_gb * count):
            infra.append(_inline_badge(f"{memory} GB"))
        if accelerator := instance_type.accelerator:
            total = accelerator.count * count
            total_text = str(int(total)) if total == int(total) else f"{total:.1f}"
            memory = accelerator.memory or state.spec_accelerator_memory
            if not memory:
                totals = _collect_metric_vals(state, "gpu_mem_total_mb")
                if totals:
                    per_gpu = sum(totals) / len(totals)
                    memory = f"{per_gpu / 1024:.0f}GB" if per_gpu >= 1024 else f"{per_gpu:.0f}MB"
            memory_text = f" {memory}" if memory else ""
            infra.append(_inline_badge(f"⚡ {total_text}× {accelerator.name}{memory_text}"))

    if state.phase == _Phase.STOPPED:
        status.append(_inline_badge("shutting down"))
        if state.tasks_done:
            status.append(_styled_badge(f"✓ {state.tasks_done} done", "done"))
        if rate := _throughput(state):
            status.append(_styled_badge(f"{rate:.1f}/min", "running"))
        if cost := _cost_badges(state):
            status.extend(cost)
        return infra, status, tasks

    total = state.desired_nodes or state.total_nodes or len(state.nodes)
    if state.phase == _Phase.PROVISIONING and total == 0:
        status.append(_inline_badge("provisioning"))
    elif total > 0:
        ssh = sum(node.value >= _NodeStatus.SSH.value for node in state.nodes.values())
        bootstrap = sum(node.value >= _NodeStatus.BOOTSTRAPPING.value for node in state.nodes.values())
        ready = sum(node == _NodeStatus.READY for node in state.nodes.values())
        status.extend((
            _progress_badge("ssh", ssh, total),
            _progress_badge("bootstrap", bootstrap, total),
            _progress_badge("ready", ready, total),
        ))

    for prefixes, label in ((("cpu",), "cpu"), (("mem",), "mem"), (("gpu_util",), "gpu")):
        values = _collect_metric_vals(state, *prefixes)
        if values:
            status.append(_gauge_inline(label, sum(values) / len(values)))
    used = _collect_metric_vals(state, "gpu_mem_mb")
    capacity = _collect_metric_vals(state, "gpu_mem_total_mb")
    if used and capacity:
        average_used = sum(used) / len(used)
        average_capacity = sum(capacity) / len(capacity)
        status.append(_gauge_inline("vram", average_used / average_capacity * 100 if average_capacity else 0))

    match state.reconciler_state:
        case "scaling_up":
            status.append(_styled_badge(f"● scaling → {state.desired_nodes}", "scaling"))
            if state.pending_nodes:
                status.append(_styled_badge(f"pending {state.pending_nodes}", "scaling"))
        case "draining":
            status.append(_styled_badge(f"● draining {state.draining_nodes}", "drifted"))
        case _ if state.phase == _Phase.READY:
            status.append(_inline_badge("in sync"))

    if state.is_elastic:
        if state.min_nodes is not None:
            status.append(_inline_badge(f"min {state.min_nodes}"))
        status.append(_styled_badge(f"cur {len(state.nodes)}", "cluster"))
        if state.max_nodes is not None:
            status.append(_inline_badge(f"max {state.max_nodes}"))
    if cost := _cost_badges(state):
        status.extend(cost)

    if state.tasks_queued or state.tasks_running or state.tasks_done or state.tasks_failed:
        if state.tasks_queued:
            tasks.append(_styled_badge(f"{state.tasks_queued} queued", "queued"))
        tasks.append(_styled_badge(f"● {state.tasks_running} running", "running"))
        tasks.append(_styled_badge(f"✓ {state.tasks_done} done", "done"))
        if state.tasks_failed:
            tasks.append(_styled_badge(f"✗ {state.tasks_failed} failed", "failed"))
        if rate := _throughput(state):
            tasks.append(_styled_badge(f"{rate:.1f} tasks/min", "running"))
            remaining = state.tasks_queued + state.tasks_running
            if remaining:
                tasks.append(_styled_badge(f"est. {_format_duration(remaining / rate * 60)}", "cost"))

    return infra, status, tasks


def _join_badges(badges: list[Text]) -> Text:
    combined = Text()
    for badge in badges:
        combined.append_text(badge)
    return combined


def _render_spinners(state: _State) -> list[Text]:
    frame = _SPINNER_FRAMES[_spinner_tick[0] % len(_SPINNER_FRAMES)]
    lines: list[Text] = []
    for node_id, timeline in state.bootstrap_spinners.items():
        line = _badge_text(_node_label(state, node_id))
        phase = _PHASE_LABELS.get(timeline.active, timeline.active) if timeline.active else "waiting"
        line.append_text(Text(f" {frame} {phase} ", style=_badge_style(timeline.active or "bootstrap")))
        if timeline.output:
            line.append(f"  {timeline.output[:80]}", style=DIM)
        lines.append(line)
    return lines


class _LiveFooter:
    def __init__(self) -> None:
        self.state = _State(total_nodes=0)

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        _spinner_tick[0] += 1
        infra, status, tasks = _collect_badges(self.state)
        spinners = _render_spinners(self.state)
        progress = [
            Text.assemble(_badge_text(_node_label(self.state, node_id)), (f"  {content}", MEDIUM))
            for node_id, content in self.state.progress_lines.items()
        ]
        parts: list[RenderableType] = [Text()]
        parts.extend(spinners)
        if spinners:
            parts.append(Text())
        parts.extend(progress)
        if progress:
            parts.append(Text())
        parts.extend(Align.center(_join_badges(group)) for group in (infra, status, tasks) if group)
        yield Group(*parts)


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    remainder = int(seconds % 60)
    return f"{minutes}m {remainder:02d}s" if minutes < 60 else f"{minutes // 60}h {minutes % 60:02d}m"


def _render_summary(state: _State, now: float | None = None) -> RenderableType:
    now = now if now is not None else time.monotonic()
    duration = now - state.pool_started_at if state.pool_started_at else 0
    overview = Table(
        title="Session Summary\n",
        title_style="bold",
        title_justify="center",
        show_header=False,
        show_edge=False,
        box=None,
        padding=(0, 2),
    )
    overview.add_column("key", style="bright_black", min_width=12)
    overview.add_column("value")
    if state.cluster:
        provider = state.cluster.spec.provider.upper()
        region = state.instances[0].region if state.instances else ""
        instance_type = state.instances[0].offer.instance_type.name if state.instances else ""
        if parts := [part for part in (provider, region, instance_type) if part]:
            overview.add_row("Provider", Text(" · ".join(parts)))
    if state.instances:
        spot = sum(1 for instance in state.instances if instance.spot)
        on_demand = len(state.instances) - spot
        allocations = [*(("$ " + str(spot) + " spot",) if spot else ()), *(("$$$ " + str(on_demand) + " on-demand",) if on_demand else ())]
        allocation = f" ({', '.join(allocations)})" if allocations else ""
        overview.add_row("Cluster", Text(f"{len(state.instances)} nodes{allocation}"))
        accelerator = state.instances[0].offer.instance_type.accelerator
        if accelerator:
            total = accelerator.count * len(state.instances)
            total_text = str(int(total)) if total == int(total) else f"{total:.1f}"
            memory = f" {accelerator.memory}" if accelerator.memory else ""
            overview.add_row("Accelerator", Text(f"{total_text}× {accelerator.name}{memory}"))
    overview.add_row("Duration", Text(_format_duration(duration)))
    hourly = sum(
        (instance.offer.spot_price if instance.spot else instance.offer.on_demand_price) or 0.0
        for instance in state.instances
    )
    if hourly > 0:
        cost = Text(f"${hourly * duration / 3600:.2f}", style="green")
        cost.append(f" (${hourly:.2f}/hr)", style="bright_black")
        overview.add_row("Cost", cost)
    task_text = Text(f"{state.tasks_done} completed", style="green bold")
    task_text.append(" · ", style="color(240)")
    task_text.append(f"{state.tasks_failed} failed", style="red bold")
    overview.add_row("Tasks", task_text)
    if rate := _throughput(state, now=now):
        overview.add_row("Throughput", Text(f"{rate:.1f} tasks/min"))
    if state.task_latencies:
        average = sum(state.task_latencies) / len(state.task_latencies)
        latency = Text(f"{average:.1f}s")
        latency.append(
            f" (min {min(state.task_latencies):.1f}s, max {max(state.task_latencies):.1f}s)",
            style="bright_black",
        )
        overview.add_row("Avg latency", latency)
    gpu = _collect_metric_vals(state, "gpu_util")
    vram = _collect_metric_vals(state, "gpu_mem_mb")
    vram_total = _collect_metric_vals(state, "gpu_mem_total_mb")
    cpu = _collect_metric_vals(state, "cpu")
    memory = _collect_metric_vals(state, "mem")
    if gpu:
        overview.add_row(
            "Avg GPU",
            Text(f"{sum(gpu) / len(gpu):.0f}% ({min(gpu):.0f}%–{max(gpu):.0f}%)"),
        )
    if vram:
        average = sum(vram) / len(vram)
        if vram_total:
            total = sum(vram_total) / len(vram_total)
            overview.add_row("Avg VRAM", Text(f"{average:.0f}/{total:.0f} MB"))
        else:
            overview.add_row("Avg VRAM", Text(f"{average:.0f} MB"))
    if cpu:
        overview.add_row("Avg CPU", Text(f"{sum(cpu) / len(cpu):.0f}%"))
    if memory:
        overview.add_row("Avg Memory", Text(f"{sum(memory) / len(memory):.0f}%"))
    breakdown = Table(
        title="Task Execution Summary\n",
        title_style="bold",
        title_justify="center",
        show_edge=False,
        box=None,
        padding=(0, 2),
        header_style="bold bright_black",
    )
    for column in ("Task", "Calls", "Avg", "Min", "Max", "Failed"):
        breakdown.add_column(column, justify="right" if column != "Task" else "left")
    for name in sorted(set(state.task_fn_stats) | set(state.task_fn_failed), key=lambda key: len(state.task_fn_stats.get(key, ())), reverse=True):
        latencies = state.task_fn_stats.get(name, ())
        failures = state.task_fn_failed.get(name, 0)
        breakdown.add_row(
            Text(name),
            Text(str(len(latencies) + failures)),
            Text(f"{sum(latencies) / len(latencies):.1f}s" if latencies else "–"),
            Text(f"{min(latencies):.1f}s" if latencies else "–", style="green"),
            Text(f"{max(latencies):.1f}s" if latencies else "–", style=WARNING_STYLE),
            Text(str(failures), style="red bold" if failures else "bright_black"),
        )
    if state.tasks_per_node:
        counts = list(state.tasks_per_node.values())
        distribution = Text(f"avg {sum(counts) / len(counts):.0f}")
        distribution.append(f" (min {min(counts)}, max {max(counts)})", style="bright_black")
        overview.add_row("Distribution", distribution)
    layout = Table.grid(padding=(0, 3), expand=True)
    layout.add_column("left", ratio=2)
    layout.add_column("right", ratio=3)
    layout.add_row(overview, breakdown)
    return Group(Text(""), layout, Text(""))
