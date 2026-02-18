"""Console actor — Rich Live display driven by spy events.

Console tells this story: idle → observing → stopped.

The skyward badge renders a fixed-step timeline with progress counters.
The cluster badge shows aggregated infrastructure info.
The metrics badge shows live avg (min–max) for every metric.
The tasks badge renders a structured table of submitted work.
Instance badges render scrolling log output.
"""

from __future__ import annotations

import math
import os
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass, replace
from enum import Enum, auto
from types import MappingProxyType

from casty import ActorContext, Behavior, Behaviors, SpyEvent, Terminated
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.style import Style
from rich.table import Table
from rich.text import Text

from skyward.api.model import Cluster, Instance
from skyward.api.spec import PoolSpec
from skyward.observability.logger import logger

from .messages import (
    BootstrapCommand,
    BootstrapConsole,
    BootstrapDone,
    BootstrapPhase,
    ClusterReady,
    Error,
    ExecuteOnNode,
    InstanceBecameReady,
    InstanceDied,
    InstancesProvisioned,
    Log,
    Metric,
    NodeBecameReady,
    NodeLost,
    PoolStarted,
    PoolStopped,
    Preempted,
    Provision,
    ShutdownRequested,
    StartPool,
    StopPool,
    SubmitBroadcast,
    SubmitTask,
    TaskResult,
    TaskSubmitted,
    _Connected,
    _ConnectionFailed,
    _LocalInstallDone,
    _PollResult,
    _PostBootstrapFailed,
    _ShutdownDone,
    _UserCodeSyncDone,
    _WorkerFailed,
    _WorkerStarted,
)


@dataclass(frozen=True, slots=True)
class LocalOutput:
    line: str
    stream: str = "stdout"


type ConsoleInput = SpyEvent | LocalOutput

def _detect_terminal_bg() -> tuple[int, int, int] | None:
    try:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
    except (OSError, termios.error):
        return None

    try:
        tty.setraw(fd)
        os.write(sys.stdout.fileno(), b'\033]11;?\033\\')

        if not select.select([fd], [], [], 0.5)[0]:
            return None

        resp = b''
        while select.select([fd], [], [], 0.1)[0]:
            resp += os.read(fd, 1024)

        decoded = resp.decode("latin-1")
        if "rgb:" not in decoded:
            return None

        rgb_part = decoded.split("rgb:")[1].split("\033")[0].split("\\")[0]
        parts = rgb_part.split("/")
        if len(parts) != 3:
            return None

        r = int(parts[0][:2], 16)
        g = int(parts[1][:2], 16)
        b = int(parts[2][:2], 16)
        return (r, g, b)
    except Exception:
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _is_dark_background() -> bool:
    bg = _detect_terminal_bg()
    if bg is None:
        return True
    r, g, b = bg
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return luminance < 128


_DARK: bool = _is_dark_background()

_BADGE_FG = "rgb(0,0,0)" if _DARK else "rgb(255,255,255)"
_BADGE_BG = "rgb(255,255,255)" if _DARK else "rgb(0,0,0)"

_FIXED_BADGES: dict[str, Style] = {
    "skyward": Style(color=_BADGE_FG, bgcolor=_BADGE_BG, bold=True),
    "tasks": Style(color=_BADGE_FG, bgcolor=_BADGE_BG),
    "cluster": Style(color=_BADGE_FG, bgcolor=_BADGE_BG),
    "logs": Style(color=_BADGE_FG, bgcolor=_BADGE_BG),
    "metrics": Style(color=_BADGE_FG, bgcolor=_BADGE_BG),
    "local": Style(color=_BADGE_FG, bgcolor=_BADGE_BG),
    "single": Style(color=_BADGE_FG, bgcolor=_BADGE_BG),
    "broadcast": Style(color=_BADGE_FG, bgcolor=_BADGE_BG),
}

_badge_registry: dict[str, Style] = {}
_badge_counter: int = 0
_GOLDEN_ANGLE = 137.508


def _badge_style(text: str) -> Style:
    if text in _FIXED_BADGES:
        return _FIXED_BADGES[text]

    if text not in _badge_registry:
        global _badge_counter
        while True:
            hue = (_badge_counter * _GOLDEN_ANGLE) % 360
            _badge_counter += 1
            if 30 < hue < 330:
                break
        r, g, b = _hsl_to_rgb(hue / 360, 0.65, 0.45)
        _badge_registry[text] = Style(color="white", bgcolor=f"rgb({r},{g},{b})")

    return _badge_registry[text]


def _hsl_to_rgb(
    h: float, s: float, lightness: float,
) -> tuple[int, int, int]:
    c = (1 - abs(2 * lightness - 1)) * s
    x = c * (1 - abs((h * 6) % 2 - 1))
    m = lightness - c / 2
    h6 = int(h * 6) % 6
    r, g, b = [(c, x, 0), (x, c, 0), (0, c, x), (0, x, c), (x, 0, c), (c, 0, x)][h6]
    return int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)


# ─── Timeline steps ──────────────────────────────────────────────


class Phase(Enum):
    INFRA = auto()
    PROVISIONED = auto()
    SSH = auto()
    BOOTSTRAP = auto()
    WORKERS = auto()
    READY = auto()


_LABELS: dict[Phase, str] = {
    Phase.INFRA: "Compute pool initialized",
    Phase.PROVISIONED: "Instances provisioned",
    Phase.SSH: "SSH connected",
    Phase.BOOTSTRAP: "Bootstrapped",
    Phase.WORKERS: "Workers joined",
    Phase.READY: "Pool ready",
}

_PHASES = tuple(Phase)


@dataclass(frozen=True, slots=True)
class _Timeline:
    total: int = 1
    done: MappingProxyType[Phase, int] = MappingProxyType({})
    active: Phase | None = None

    def advance(self, phase: Phase, count: int = 1) -> _Timeline:
        current = self.done.get(phase, 0)
        new_done = MappingProxyType({**self.done, phase: current + count})
        new_active = phase if new_done[phase] < self.total else self.active
        return replace(self, done=new_done, active=new_active)

    def complete(self, phase: Phase) -> _Timeline:
        new_done = MappingProxyType({**self.done, phase: self.total})
        return replace(self, done=new_done)

    def set_active(self, phase: Phase) -> _Timeline:
        return replace(self, active=phase)


def _glow_style() -> Style:
    t = time.monotonic()
    brightness = 0.5 + 0.5 * math.sin(t * 3.0)
    if _DARK:
        lo, hi = 100, 255
    else:
        lo, hi = 0, 140
    v = int(lo + (hi - lo) * brightness)
    return Style(color=f"rgb({v},{v},{v})", bold=True)


def _render_timeline(timeline: _Timeline) -> list[Text]:
    lines: list[Text] = []
    for phase in _PHASES:
        count = timeline.done.get(phase, 0)
        label = _LABELS[phase]

        needs_counter = phase not in (Phase.INFRA, Phase.READY)
        if needs_counter and timeline.total > 1:
            label = f"{label} ({count}/{timeline.total})"

        in_progress = 0 < count < timeline.total
        if count >= timeline.total:
            lines.append(Text(f"  ✓ {label}", style="green"))
        elif in_progress or phase == timeline.active:
            lines.append(Text(f"  ◆ {label}", style=_glow_style()))
        else:
            lines.append(Text(f"  ○ {label}", style="dim"))

    return lines


# ─── State ────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _LogEntry:
    instance_id: str
    line: str


class _TaskStatus(Enum):
    RUNNING = auto()
    DONE = auto()


@dataclass(frozen=True, slots=True)
class _TaskEntry:
    task_id: str
    fn: object
    args: tuple
    kwargs: dict[str, object]
    kind: str
    status: _TaskStatus = _TaskStatus.RUNNING
    started_at: float = 0.0
    elapsed: float | None = None
    broadcast_done: int = 0
    instance_id: str = ""


@dataclass(frozen=True, slots=True)
class _MetricSample:
    value: float
    timestamp: float


@dataclass(frozen=True, slots=True)
class _State:
    timeline: _Timeline = _Timeline()
    tasks: MappingProxyType[str, _TaskEntry] = MappingProxyType({})
    task_order: tuple[str, ...] = ()
    log: tuple[_LogEntry, ...] = ()
    cluster: Cluster | None = None
    instances: tuple[Instance, ...] = ()
    metrics: MappingProxyType[str, MappingProxyType[str, _MetricSample]] = MappingProxyType({})
    pool_started_at: float = 0.0
    console: Console = Console(stderr=True)
    live: Live | None = None


_MAX_LOG_ENTRIES = 100


def _inst(state: _State, instance_id: str, line: str) -> _State:
    entry = _LogEntry(instance_id=instance_id, line=line)
    log = (*state.log, entry)
    if len(log) > _MAX_LOG_ENTRIES:
        log = log[-_MAX_LOG_ENTRIES:]
    return replace(state, log=log)


# ─── Rendering ────────────────────────────────────────────────────


def _render_skyward_col(state: _State) -> Group:
    lines: list[Text] = []
    lines.append(Text(" skyward ", style=_badge_style("skyward")))
    lines.extend(_render_timeline(state.timeline))
    return Group(*lines)


def _render_cluster_col(state: _State) -> Group:
    lines: list[RenderableType] = []
    lines.append(Text(" cluster ", style=_badge_style("cluster")))

    insts = state.instances
    if not insts:
        lines.append(Text("  waiting for instances…", style="dim"))
        return Group(*lines)

    first = insts[0]
    provider_name = (state.cluster.spec.provider or "").upper() if state.cluster else ""
    region = first.region or ""
    if provider_name and region:
        lines.append(Text(f"  {provider_name} · {region}", style="dim"))
    elif provider_name or region:
        lines.append(Text(f"  {provider_name}{region}", style="dim"))

    n_spot = sum(1 for i in insts if i.spot)
    n_od = len(insts) - n_spot
    node_parts: list[str] = []
    if n_spot:
        node_parts.append(f"{n_spot} spot")
    if n_od:
        node_parts.append(f"{n_od} OD")
    node_detail = f" ({', '.join(node_parts)})" if node_parts else ""
    lines.append(Text(f"  {len(insts)} nodes{node_detail}"))

    if first.instance_type:
        lines.append(Text(f"  {first.instance_type}", style="dim"))

    if first.gpu_model:
        total_gpu = sum(i.gpu_count for i in insts)
        vram_total = sum(i.gpu_vram_gb * i.gpu_count for i in insts)
        gpu_label = f"  {total_gpu}× {first.gpu_model}"
        if vram_total:
            gpu_label += f" ({vram_total} GB)"
        lines.append(Text(gpu_label))

    total_vcpus = sum(i.vcpus for i in insts)
    total_ram = sum(i.memory_gb for i in insts)
    if total_vcpus or total_ram:
        hw_parts: list[str] = []
        if total_vcpus:
            hw_parts.append(f"{total_vcpus} vCPUs")
        if total_ram:
            hw_parts.append(f"{total_ram:.0f} GB RAM")
        lines.append(Text(f"  {' · '.join(hw_parts)}"))

    hourly = sum(i.hourly_rate for i in insts)
    if hourly > 0:
        started = state.pool_started_at
        elapsed_h = (time.monotonic() - started) / 3600 if started else 0
        total_cost = hourly * elapsed_h
        lines.append(Text(f"  ${hourly:.2f}/hr · ${total_cost:.2f} total", style="dim"))

    return Group(*lines)




def _render_tasks_col(state: _State, width: int) -> Group:
    lines: list[RenderableType] = []
    badge = Text(" tasks ", style=_badge_style("tasks"))
    done = sum(1 for t in state.tasks.values() if t.status == _TaskStatus.DONE)
    badge.append(f"  {done}/{len(state.tasks)}", style="dim")
    lines.append(badge)

    max_sig = int(width * 0.5)
    visible = list(state.task_order[-8:])
    n_instances = len(state.instances)

    for tid in visible:
        entry = state.tasks[tid]
        line = Text("  ")
        name = _format_task(entry.fn, entry.args, entry.kwargs, max_sig=max_sig)

        match entry.status:
            case _TaskStatus.RUNNING:
                elapsed = time.monotonic() - entry.started_at
                line.append("◆ ", style=_glow_style())
                line.append(name)
                if entry.kind == "broadcast" and n_instances > 0:
                    line.append(f"  ({entry.broadcast_done}/{n_instances})", style="dim")
                line.append(f"  {elapsed:.1f}s", style=_glow_style())
                if entry.instance_id:
                    line.append("  ")
                    line.append(f" {entry.instance_id[:8]} ", style=_badge_style(entry.instance_id))
            case _TaskStatus.DONE:
                duration = f"  {entry.elapsed:.1f}s" if entry.elapsed is not None else ""
                line.append("✓ ", style="green")
                line.append(name)
                line.append(duration, style="green")
                if entry.kind == "broadcast":
                    line.append("  ")
                    line.append(" broadcast ", style=_badge_style("broadcast"))
                elif entry.instance_id:
                    line.append("  ")
                    line.append(f" {entry.instance_id[:8]} ", style=_badge_style(entry.instance_id))

        lines.append(line)

    return Group(*lines)


def _full_width_badge(label: str, width: int, suffix: str = "") -> Text:
    badge = Text(f" {label} ", style=_badge_style(label))
    if suffix:
        badge.append(suffix, style=_badge_style(label))
    used = len(label) + 2 + len(suffix)
    remaining = width - used
    if remaining > 0:
        badge.append(" " * remaining, style=_badge_style(label))
    return badge


_BAR_WIDTH = 12
_BAR_FILLED = "▓"
_BAR_EMPTY = "░"
_METRICS_PAGE_INTERVAL = 5.0
_METRICS_MAX_INSTANCES = 6


def _progress_bar(pct: float, width: int = _BAR_WIDTH) -> Text:
    filled = int(round(pct / 100 * width))
    filled = max(0, min(width, filled))
    bar = Text(_BAR_FILLED * filled, style="bold")
    bar.append(_BAR_EMPTY * (width - filled), style="dim")
    return bar


@dataclass(frozen=True, slots=True)
class _InstanceMetrics:
    instance_id: str
    cpu: float | None = None
    mem_pct: float | None = None
    mem_used_mb: float | None = None
    mem_total_mb: float | None = None
    gpu_util: float | None = None
    gpu_mem_used_mb: float | None = None
    gpu_mem_total_mb: float | None = None
    gpu_temp: float | None = None


def _collect_instance_metrics(
    instance_id: str, raw: MappingProxyType[str, _MetricSample],
) -> _InstanceMetrics:
    def _get(*names: str) -> float | None:
        for name in names:
            for key, sample in raw.items():
                if key == name or key.startswith(f"{name}_"):
                    return sample.value
        return None

    mem_used = _get("mem_used_mb")
    mem_total = _get("mem_total_mb")
    mem_pct = (mem_used / mem_total * 100) if mem_used is not None and mem_total else None

    return _InstanceMetrics(
        instance_id=instance_id,
        cpu=_get("cpu"),
        mem_pct=mem_pct,
        mem_used_mb=mem_used,
        mem_total_mb=mem_total,
        gpu_util=_get("gpu_util"),
        gpu_mem_used_mb=_get("gpu_mem_mb"),
        gpu_mem_total_mb=_get("gpu_mem_total_mb"),
        gpu_temp=_get("gpu_temp"),
    )


def _format_mem(used_mb: float | None, total_mb: float | None) -> str:
    if used_mb is None or total_mb is None:
        return ""
    if total_mb >= 1024:
        return f"({used_mb / 1024:.1f}/{total_mb / 1024:.1f} GB)"
    return f"({used_mb:.0f}/{total_mb:.0f} MB)"


def _render_instance_metrics_line(m: _InstanceMetrics) -> Text:
    line = Text()
    short_id = m.instance_id[:8].center(8)
    line.append(f" {short_id} ", style=_badge_style(m.instance_id))

    parts: list[tuple[str, float | None, str]] = []

    if m.cpu is not None:
        mem_info = _format_mem(m.mem_used_mb, m.mem_total_mb)
        parts.append(("cpu", m.cpu, ""))
        parts.append(("memory", m.mem_pct, mem_info))

    if m.gpu_util is not None:
        gpu_mem_info = _format_mem(m.gpu_mem_used_mb, m.gpu_mem_total_mb)
        parts.append(("gpu utilization", m.gpu_util, ""))
        parts.append(("gpu memory", None, gpu_mem_info))

    if m.gpu_temp is not None:
        parts.append(("gpu temperature", None, f"{m.gpu_temp:.0f}°"))

    for i, (label, pct, extra) in enumerate(parts):
        sep = "   " if i > 0 else "  "
        line.append(sep, style="dim")
        line.append(f"{label} ", style="dim")
        if pct is not None:
            line.append_text(_progress_bar(pct))
            line.append(f" {pct:.1f}%")
        if extra:
            line.append(f" {extra}", style="dim")

    return line


def _render_metrics_footer(state: _State, width: int) -> Group:
    lines: list[RenderableType] = []

    if not state.metrics:
        return Group()

    instance_ids = sorted(state.metrics.keys())
    total = len(instance_ids)
    page_size = min(_METRICS_MAX_INSTANCES, total)
    total_pages = math.ceil(total / page_size) if page_size else 1

    if total_pages > 1:
        page_idx = int(time.monotonic() / _METRICS_PAGE_INTERVAL) % total_pages
        start = page_idx * page_size
        visible_ids = instance_ids[start:start + page_size]
        suffix = f"({page_idx + 1}/{total_pages})"
    else:
        visible_ids = instance_ids
        suffix = ""

    lines.append(_full_width_badge("metrics", width, f" {suffix}" if suffix else ""))

    for iid in visible_ids:
        raw = state.metrics[iid]
        m = _collect_instance_metrics(iid, raw)
        lines.append(_render_instance_metrics_line(m))

    return Group(*lines)


def _measure_height(renderable: RenderableType, width: int) -> int:
    measure_console = Console(width=width, file=open(os.devnull, "w"))  # noqa: SIM115
    with measure_console.capture() as capture:
        measure_console.print(renderable, end="")
    return capture.get().count("\n") + 1


_SKYWARD_COL_WIDTH = 35


def _render(state: _State) -> RenderableType:
    parts: list[RenderableType] = []
    term_w = state.console.size.width
    left_width = _SKYWARD_COL_WIDTH + 4 + _SKYWARD_COL_WIDTH
    tasks_width = max(20, term_w - left_width - 4)
    cluster_width = _SKYWARD_COL_WIDTH

    left_top = Table.grid(padding=(0, 4))
    left_top.add_column(width=_SKYWARD_COL_WIDTH)
    left_top.add_column(width=cluster_width)
    left_top.add_row(
        _render_skyward_col(state),
        _render_cluster_col(state),
    )

    header = Table.grid(padding=(0, 4))
    header.add_column(width=left_width)
    header.add_column(width=tasks_width)
    header.add_row(left_top, _render_tasks_col(state, tasks_width))
    parts.append(header)

    parts.append(Text())
    parts.append(_full_width_badge("logs", term_w))

    metrics_footer = _render_metrics_footer(state, term_w)
    footer_height = _measure_height(metrics_footer, term_w) + 1  # +1 blank line

    term_h = state.console.size.height
    header_height = _measure_height(Group(*parts), term_w)
    budget = max(1, term_h - header_height - footer_height)
    visible = state.log[-budget:]

    for entry in visible:
        line = Text()
        short_id = entry.instance_id[:8].center(8)
        line.append(f" {short_id} ", style=_badge_style(entry.instance_id))
        line.append(f"  {entry.line}")
        parts.append(line)

    used_log_lines = len(visible)
    padding = budget - used_log_lines
    for _ in range(padding):
        parts.append(Text())

    parts.append(Text())
    parts.append(metrics_footer)

    return Group(*parts)


def _format_task(fn: object, args: tuple, kwargs: dict, max_sig: int = 40) -> str:
    name = getattr(fn, "__name__", str(fn))
    parts = [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
    sig = ", ".join(parts)
    if len(sig) > max_sig:
        return f"{name}(…)"
    return f"{name}({sig})" if sig else name


# ─── Actor ────────────────────────────────────────────────────────


def console_actor(spec: PoolSpec) -> Behavior[ConsoleInput]:
    """Console tells this story: idle → observing → stopped."""

    def idle() -> Behavior[ConsoleInput]:
        async def setup(ctx: ActorContext[ConsoleInput]) -> Behavior[ConsoleInput]:
            timeline = _Timeline(total=spec.nodes)
            return observing(_State(timeline=timeline))
        return Behaviors.setup(setup)

    def observing(state: _State) -> Behavior[ConsoleInput]:
        async def receive(
            ctx: ActorContext[ConsoleInput], msg: ConsoleInput,
        ) -> Behavior[ConsoleInput]:
            match msg:
                case SpyEvent(event=Terminated()):
                    return Behaviors.same()

                # ─── Pool lifecycle ───────────────────────────────
                case SpyEvent(event=StartPool()):
                    tl = state.timeline.set_active(Phase.INFRA)
                    return _refresh(replace(state, timeline=tl, pool_started_at=time.monotonic()))

                case SpyEvent(event=ClusterReady()):
                    tl = state.timeline.complete(Phase.INFRA).set_active(Phase.PROVISIONED)
                    return _refresh(replace(state, timeline=tl))

                case SpyEvent(event=InstancesProvisioned(cluster=cluster, instances=raw)):
                    tl = state.timeline.complete(Phase.PROVISIONED).set_active(Phase.SSH)
                    return _refresh(replace(
                        state, timeline=tl,
                        cluster=cluster, instances=tuple(raw),
                    ))

                case SpyEvent(event=PoolStarted()):
                    tl = state.timeline.complete(Phase.READY)
                    return _refresh(replace(state, timeline=tl))

                case SpyEvent(event=StopPool()):
                    if state.live is not None:
                        state.live.stop()
                    return observing(replace(state, live=None))

                case SpyEvent(event=PoolStopped() | _ShutdownDone()):
                    return Behaviors.same()

                # ─── Node lifecycle ───────────────────────────────
                case SpyEvent(event=Provision()):
                    return Behaviors.same()

                case SpyEvent(event=NodeBecameReady()):
                    return Behaviors.same()

                case SpyEvent(event=NodeLost(node_id=nid, reason=reason)):
                    return _refresh(_inst(state, "skyward", f"Node {nid} lost: {reason}"))

                # ─── Instance lifecycle ───────────────────────────
                case SpyEvent(event=_PollResult()):
                    return Behaviors.same()

                case SpyEvent(event=_Connected()):
                    tl = state.timeline.advance(Phase.SSH)
                    return _refresh(replace(state, timeline=tl))

                case SpyEvent(event=_ConnectionFailed(error=error)):
                    return _refresh(_inst(state, "skyward", f"SSH failed: {error}"))

                case SpyEvent(event=InstanceBecameReady()):
                    return Behaviors.same()

                case SpyEvent(event=InstanceDied(instance_id=iid, reason=reason)):
                    return _refresh(_inst(state, iid, f"Died: {reason}"))

                case SpyEvent(event=Preempted(reason=reason)):
                    return _refresh(_inst(state, "skyward", f"Preempted: {reason}"))

                # ─── Bootstrap & post-bootstrap ───────────────────
                case SpyEvent(event=BootstrapConsole() as ev):
                    content = ev.content.strip()
                    if not content or content.startswith("#"):
                        return Behaviors.same()
                    return _refresh(_inst(state, ev.instance.id, content[:120]))

                case SpyEvent(event=BootstrapPhase() as ev):
                    match ev.event:
                        case "started":
                            tl = state.timeline.set_active(Phase.BOOTSTRAP)
                            new_state = replace(state, timeline=tl)
                            line = f"▸ {ev.phase}..."
                        case "completed":
                            elapsed_str = f" ({ev.elapsed:.1f}s)" if ev.elapsed else ""
                            new_state = state
                            line = f"✓ {ev.phase}{elapsed_str}"
                        case "failed":
                            new_state = state
                            line = f"✗ {ev.phase}: {ev.error}"
                        case _:
                            return Behaviors.same()
                    return _refresh(_inst(new_state, ev.instance.id, line))

                case SpyEvent(event=BootstrapCommand() as ev):
                    cmd = ev.command.strip()
                    if not cmd:
                        return Behaviors.same()
                    display = f"$ {cmd[:80]}..." if len(cmd) > 80 else f"$ {cmd}"
                    return _refresh(_inst(state, ev.instance.id, display))

                case SpyEvent(event=BootstrapDone(instance=inst, success=ok, error=err)):
                    if ok:
                        tl = state.timeline.advance(Phase.BOOTSTRAP)
                        return _refresh(replace(state, timeline=tl))
                    return _refresh(_inst(state, inst.id, f"Bootstrap failed: {err}"))

                case SpyEvent(event=_LocalInstallDone() | _UserCodeSyncDone()):
                    return Behaviors.same()

                case SpyEvent(event=_PostBootstrapFailed(error=err)):
                    return _refresh(_inst(state, "skyward", f"Post-bootstrap failed: {err}"))

                case SpyEvent(event=_WorkerStarted()):
                    tl = state.timeline.advance(Phase.WORKERS)
                    return _refresh(replace(state, timeline=tl))

                case SpyEvent(event=_WorkerFailed(error=error)):
                    return _refresh(_inst(state, "skyward", f"Worker failed: {error}"))

                # ─── Runtime ──────────────────────────────────────
                case SpyEvent(event=Log() as ev):
                    line = ev.line.strip()
                    if not line:
                        return Behaviors.same()
                    return _refresh(_inst(state, ev.instance.id, line[:120]))

                case SpyEvent(event=SubmitTask(task_id=tid) as ev) if tid not in state.tasks:
                    entry = _TaskEntry(
                        task_id=tid, fn=ev.fn, args=ev.args,
                        kwargs=ev.kwargs, kind="single",
                        started_at=time.monotonic(),
                    )
                    new_tasks = MappingProxyType({**state.tasks, tid: entry})
                    new_order = (*state.task_order, tid)
                    return _refresh(replace(state, tasks=new_tasks, task_order=new_order))

                case SpyEvent(event=SubmitBroadcast(task_id=tid) as ev) if tid not in state.tasks:
                    entry = _TaskEntry(
                        task_id=tid, fn=ev.fn, args=ev.args,
                        kwargs=ev.kwargs, kind="broadcast",
                        started_at=time.monotonic(),
                    )
                    new_tasks = MappingProxyType({**state.tasks, tid: entry})
                    new_order = (*state.task_order, tid)
                    return _refresh(replace(state, tasks=new_tasks, task_order=new_order))

                case SpyEvent(event=SubmitTask() | SubmitBroadcast()):
                    return Behaviors.same()

                case SpyEvent(event=TaskSubmitted(task_id=tid, node_id=nid)):
                    logger.warning(
                        "CONSOLE: TaskSubmitted tid={tid} nid={nid} known={known}",
                        tid=tid, nid=nid, known=tid in state.tasks,
                    )
                    if tid in state.tasks:
                        iid = state.instances[nid].id if nid < len(state.instances) else ""
                        updated = replace(state.tasks[tid], instance_id=iid)
                        new_tasks = MappingProxyType({**state.tasks, tid: updated})
                        return _refresh(replace(state, tasks=new_tasks))
                    return Behaviors.same()

                case SpyEvent(event=TaskResult(node_id=nid)):
                    first_running = next(
                        (
                            tid for tid in state.task_order
                            if state.tasks[tid].status == _TaskStatus.RUNNING
                        ),
                        None,
                    )
                    if first_running is None:
                        return Behaviors.same()
                    entry = state.tasks[first_running]
                    elapsed = time.monotonic() - entry.started_at
                    iid = state.instances[nid].id if nid < len(state.instances) else ""
                    match entry.kind:
                        case "broadcast":
                            new_done = entry.broadcast_done + 1
                            if new_done >= len(state.instances):
                                updated = replace(
                                    entry, status=_TaskStatus.DONE,
                                    elapsed=elapsed, broadcast_done=new_done,
                                )
                            else:
                                updated = replace(entry, broadcast_done=new_done)
                        case _:
                            updated = replace(
                                entry, status=_TaskStatus.DONE,
                                elapsed=elapsed, instance_id=iid,
                            )
                    new_tasks = MappingProxyType({**state.tasks, first_running: updated})
                    return _refresh(replace(state, tasks=new_tasks))

                case SpyEvent(event=Metric() as ev):
                    iid = ev.instance.id
                    sample = _MetricSample(value=ev.value, timestamp=ev.timestamp)
                    inst_metrics = dict(state.metrics.get(iid, MappingProxyType({})))
                    inst_metrics[ev.name] = sample
                    new_metrics = MappingProxyType({
                        **state.metrics,
                        iid: MappingProxyType(inst_metrics),
                    })
                    return _refresh(replace(state, metrics=new_metrics))

                case SpyEvent(event=ExecuteOnNode()):
                    return Behaviors.same()

                case SpyEvent(event=Error() as ev):
                    return _refresh(_inst(state, "skyward", f"ERROR: {ev.message}"))

                case SpyEvent(event=ShutdownRequested()):
                    if state.live is not None:
                        state.live.stop()
                    return observing(replace(state, live=None))

                case LocalOutput(line=line):
                    stripped = line.rstrip()
                    if not stripped:
                        return Behaviors.same()
                    return _refresh(_inst(state, "local", stripped))

                case _:
                    return Behaviors.same()

        return Behaviors.receive(receive)

    class _LiveRenderable:
        def __init__(self, state: _State) -> None:
            self.state = state

        def __rich__(self) -> RenderableType:
            return _render(self.state)

    def _refresh(new_state: _State) -> Behavior[ConsoleInput]:
        renderable = _LiveRenderable(new_state)
        match new_state.live:
            case None:
                live = Live(
                    renderable, console=new_state.console,
                    refresh_per_second=60, screen=False,
                )
                live.start()
                return observing(replace(new_state, live=live))
            case live:
                live.update(renderable)
                return observing(new_state)

    return idle()
