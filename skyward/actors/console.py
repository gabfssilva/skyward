"""Console actor — streaming log + adaptive footer driven by spy events.

Console tells this story: idle → observing → stopped.

Architecture (MVC in a single file):
- Model:      frozen state types + pure transition functions
- View:       badge styling, stream emitters, footer/summary renderers
- Controller: Casty behavior that receives SpyEvents, updates model, calls view
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from enum import Enum, auto
from types import MappingProxyType
from typing import Any

from casty import ActorContext, Behavior, Behaviors, SpyEvent, Terminated
from rich.align import Align
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.style import Style
from rich.table import Table
from rich.text import Text

from skyward.api.model import Cluster, Instance
from skyward.api.spec import PoolSpec

from .messages import (
    BootstrapCommand,
    BootstrapConsole,
    BootstrapDone,
    BootstrapPhase,
    ClusterReady,
    DesiredCountChanged,
    DrainComplete,
    DrainNode,
    Error,
    ExecuteOnNode,
    InstancesProvisioned,
    Log,
    Metric,
    NodeBecameReady,
    NodeJoined,
    NodeLost,
    PoolStarted,
    PoolStopped,
    Preempted,
    Provision,
    ReconcilerNodeLost,
    ShutdownRequested,
    SpawnNodes,
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

# =============================================================================
# Model
# =============================================================================


class _Phase(Enum):
    PROVISIONING = auto()
    SSH = auto()
    BOOTSTRAP = auto()
    WORKERS = auto()
    READY = auto()
    STOPPING = auto()


class _NodeStatus(Enum):
    WAITING = auto()
    SSH = auto()
    BOOTSTRAPPING = auto()
    READY = auto()


@dataclass(frozen=True, slots=True)
class _TaskEntry:
    task_id: str
    name: str
    kind: str
    started_at: float
    instance_id: str = ""
    broadcast_total: int = 0
    broadcast_done: int = 0


@dataclass(frozen=True, slots=True)
class _State:
    total_nodes: int
    phase: _Phase = _Phase.PROVISIONING
    nodes: MappingProxyType[str, _NodeStatus] = MappingProxyType({})
    tasks_queued: int = 0
    tasks_running: int = 0
    tasks_done: int = 0
    tasks_failed: int = 0
    first_task_at: float = 0.0
    cluster: Cluster | None = None
    instances: tuple[Instance, ...] = ()
    metrics: MappingProxyType[str, MappingProxyType[str, float]] = MappingProxyType({})
    pool_started_at: float = 0.0
    task_latencies: tuple[float, ...] = ()
    inflight: MappingProxyType[str, _TaskEntry] = MappingProxyType({})
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
    tasks_per_instance: MappingProxyType[str, int] = MappingProxyType({})


# --- State transitions (pure) ---


def _on_start_pool(state: _State) -> _State:
    return replace(state, phase=_Phase.PROVISIONING, pool_started_at=time.monotonic())


def _on_cluster_ready(state: _State) -> _State:
    return replace(state, phase=_Phase.SSH)


def _on_instances_provisioned(
    state: _State, cluster: Cluster, instances: tuple[Instance, ...],
) -> _State:
    return replace(state, cluster=cluster, instances=instances)


def _advance(current: _Phase, candidate: _Phase) -> _Phase:
    return candidate if candidate.value > current.value else current


def _on_ssh_connected(state: _State, instance_id: str) -> _State:
    nodes = MappingProxyType({**state.nodes, instance_id: _NodeStatus.SSH})
    ssh_count = sum(1 for s in nodes.values() if s.value >= _NodeStatus.SSH.value)
    phase = _advance(state.phase, _Phase.BOOTSTRAP) if ssh_count >= state.total_nodes else state.phase
    return replace(state, nodes=nodes, phase=phase)


def _on_bootstrap_done(state: _State, instance_id: str) -> _State:
    nodes = MappingProxyType({**state.nodes, instance_id: _NodeStatus.BOOTSTRAPPING})
    done = sum(1 for s in nodes.values() if s.value >= _NodeStatus.BOOTSTRAPPING.value)
    phase = _advance(state.phase, _Phase.WORKERS) if done >= state.total_nodes else state.phase
    return replace(state, nodes=nodes, phase=phase)


def _on_worker_started(state: _State, instance_id: str) -> _State:
    nodes = MappingProxyType({**state.nodes, instance_id: _NodeStatus.READY})
    ready = sum(1 for s in nodes.values() if s.value >= _NodeStatus.READY.value)
    phase = _advance(state.phase, _Phase.READY) if ready >= state.total_nodes else state.phase
    ready_at = time.monotonic() if phase == _Phase.READY and not state.ready_at else state.ready_at
    return replace(state, nodes=nodes, phase=phase, ready_at=ready_at)


def _on_task_submitted(
    state: _State, task_id: str, name: str, kind: str,
) -> _State:
    entry = _TaskEntry(
        task_id=task_id, name=name, kind=kind, started_at=time.monotonic(),
        broadcast_total=len(state.instances) if kind == "broadcast" else 0,
    )
    first = state.first_task_at or time.monotonic()
    inflight = MappingProxyType({**state.inflight, task_id: entry})
    return replace(
        state, tasks_queued=state.tasks_queued + 1,
        first_task_at=first, inflight=inflight,
    )


def _on_task_assigned(state: _State, task_id: str, instance_id: str) -> _State:
    if task_id not in state.inflight:
        return state
    entry = state.inflight[task_id]
    already_assigned = bool(entry.instance_id)
    started = time.monotonic() if not entry.instance_id else entry.started_at
    updated = replace(entry, instance_id=instance_id, started_at=started)
    return replace(
        state,
        tasks_queued=max(0, state.tasks_queued - 1) if not already_assigned else state.tasks_queued,
        tasks_running=state.tasks_running + 1 if not already_assigned else state.tasks_running,
        inflight=MappingProxyType({**state.inflight, task_id: updated}),
    )


def _on_task_done(state: _State, task_id: str, elapsed: float) -> _State:
    entry = state.inflight.get(task_id)
    fn_name = entry.name.split("(")[0] if entry else "unknown"
    fn_stats = {**state.task_fn_stats}
    fn_stats[fn_name] = (*fn_stats.get(fn_name, ()), elapsed)
    inflight = dict(state.inflight)
    inflight.pop(task_id, None)
    per_inst = dict(state.tasks_per_instance)
    if entry and entry.instance_id:
        per_inst[entry.instance_id] = per_inst.get(entry.instance_id, 0) + 1
    return replace(
        state, tasks_running=max(0, state.tasks_running - 1),
        tasks_done=state.tasks_done + 1,
        task_latencies=(*state.task_latencies, elapsed),
        inflight=MappingProxyType(inflight),
        task_fn_stats=MappingProxyType(fn_stats),
        tasks_per_instance=MappingProxyType(per_inst),
    )


def _on_task_failed(state: _State, task_id: str) -> _State:
    entry = state.inflight.get(task_id)
    fn_name = entry.name.split("(")[0] if entry else "unknown"
    fn_failed = {**state.task_fn_failed}
    fn_failed[fn_name] = fn_failed.get(fn_name, 0) + 1
    inflight = dict(state.inflight)
    inflight.pop(task_id, None)
    return replace(
        state, tasks_running=max(0, state.tasks_running - 1),
        tasks_failed=state.tasks_failed + 1,
        inflight=MappingProxyType(inflight),
        task_fn_failed=MappingProxyType(fn_failed),
    )


def _on_broadcast_partial(state: _State, task_id: str) -> _State:
    if task_id not in state.inflight:
        return state
    entry = state.inflight[task_id]
    updated = replace(entry, broadcast_done=entry.broadcast_done + 1)
    return replace(state, inflight=MappingProxyType({**state.inflight, task_id: updated}))


def _on_metric(state: _State, instance_id: str, name: str, value: float) -> _State:
    inst_metrics = dict(state.metrics.get(instance_id, MappingProxyType({})))
    inst_metrics[name] = value
    new_metrics = MappingProxyType({
        **state.metrics, instance_id: MappingProxyType(inst_metrics),
    })
    return replace(state, metrics=new_metrics)


def _on_desired_changed(state: _State, desired: int) -> _State:
    match desired:
        case d if d > state.desired_nodes:
            reconciler = "scaling_up"
        case d if d < len(state.nodes):
            reconciler = "draining"
        case _:
            reconciler = "watching"
    return replace(state, desired_nodes=desired, reconciler_state=reconciler)


def _on_spawn_nodes(
    state: _State, instances: tuple[Instance, ...], cluster: Cluster | None = None,
) -> _State:
    return replace(
        state,
        pending_nodes=state.pending_nodes + len(instances),
        instances=(*state.instances, *instances),
        total_nodes=state.total_nodes + len(instances),
        cluster=cluster or state.cluster,
    )


def _on_node_joined(state: _State) -> _State:
    pending = max(0, state.pending_nodes - 1)
    reconciler = "watching" if pending == 0 and state.draining_nodes == 0 else state.reconciler_state
    return replace(state, pending_nodes=pending, reconciler_state=reconciler)


def _on_drain_node(state: _State) -> _State:
    return replace(state, draining_nodes=state.draining_nodes + 1, reconciler_state="draining")


def _on_drain_complete(state: _State, instance_id: str = "") -> _State:
    draining = max(0, state.draining_nodes - 1)
    reconciler = "watching" if draining == 0 else state.reconciler_state
    nodes = state.nodes
    if instance_id and instance_id in nodes:
        nodes = MappingProxyType({k: v for k, v in nodes.items() if k != instance_id})
    return replace(
        state,
        draining_nodes=draining,
        reconciler_state=reconciler,
        nodes=nodes,
        total_nodes=max(0, state.total_nodes - 1),
    )


def _on_reconciler_node_lost(state: _State) -> _State:
    pending = max(0, state.pending_nodes - 1)
    return replace(state, pending_nodes=pending)


def _throughput(state: _State, now: float | None = None) -> float:
    if not state.tasks_done:
        return 0.0
    ts = now if now is not None else time.monotonic()
    elapsed_min = (ts - state.first_task_at) / 60
    return state.tasks_done / elapsed_min if elapsed_min > 0 else 0.0


# =============================================================================
# View
# =============================================================================

_LOGO_LINES = (
"   ▌           ▌",
" ▛▘▙▘▌▌▌▌▌▀▌▛▘▛▌",
" ▄▌▛▖▙▌▚▚▘█▌▌ ▙▌",
"     ▄▌")

# --- Styles ---

DIM = Style(color="bright_black")
MEDIUM = Style(color="white")
BRIGHT = Style(bold=True)
GREEN = Style(color="green", bold=True)
YELLOW = Style(color="yellow", bold=True)
CYAN = Style(color="cyan", bold=True)
RED = Style(color="red", bold=True)
VERY_DIM = Style(color="color(240)")
COST_DIM = Style(color="color(245)")


# --- Terminal theme detection ---


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

        resp = b""
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


_DARK = _is_dark_background()
_BADGE_L = 0.55 if _DARK else 0.35
_BADGE_FG = "rgb(0,0,0)" if _DARK else "rgb(255,255,255)"


# --- Badge styling ---

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


def _hsl_to_rgb(h: float, s: float, lightness: float) -> tuple[int, int, int]:
    c = (1 - abs(2 * lightness - 1)) * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = lightness - c / 2
    if h < 60:
        r, g, b = c, x, 0.0
    elif h < 120:
        r, g, b = x, c, 0.0
    elif h < 180:
        r, g, b = 0.0, c, x
    elif h < 240:
        r, g, b = 0.0, x, c
    elif h < 300:
        r, g, b = x, 0.0, c
    else:
        r, g, b = c, 0.0, x
    return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))


def _make_badge(hue: float, saturation: float) -> Style:
    r, g, b = _hsl_to_rgb(hue % 360, saturation, _BADGE_L)
    return Style(color=_BADGE_FG, bgcolor=f"rgb({r},{g},{b})", bold=True)


_FIXED_BADGES: dict[str, Style] = {
    name: _make_badge(hue, sat) for name, (hue, sat) in _FIXED_BADGE_HUES.items()
}
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
    hue = (_stable_hash(label) * 137.508) % 360
    return _make_badge(hue, 0.65)


def _gauge_badge(_label: str, pct: float) -> Style:
    hue = 120.0 * (1 - min(pct, 100.0) / 100.0)
    return _make_badge(hue, 0.55)


# --- Stream emitters ---


def _badge_text(label: str) -> Text:
    short = label[:8].center(8) if len(label) > 8 else label.center(8)
    t = Text()
    t.append(f" {short} ", style=_badge_style(label))
    return t


def _inline_badge(label: str) -> Text:
    t = Text()
    t.append(f" {label} ", style=_badge_style(label))
    return t


def _emit(console: Console, badge: str, text: str, style: str = "") -> None:
    line = _badge_text(badge)
    line.append(f"  {text}", style=style or None)
    console.print(line)


def _emit_task(console: Console, badge: str, status: str, text: str) -> None:
    line = _badge_text(badge)
    line.append(" ")
    line.append_text(_inline_badge(status))
    line.append(f" {text}")
    console.print(line)


# --- Metric helpers ---


def _find_metric(
    raw: MappingProxyType[str, float], *prefixes: str,
) -> float | None:
    for prefix in prefixes:
        if prefix in raw:
            return raw[prefix]
        for key, val in raw.items():
            if key.startswith(f"{prefix}_"):
                return val
    return None


def _collect_metric_vals(
    state: _State, *prefixes: str,
) -> list[float]:
    vals: list[float] = []
    for iid in state.metrics:
        if (v := _find_metric(state.metrics[iid], *prefixes)) is not None:
            vals.append(v)
    return vals


def _cost_badges(state: _State) -> tuple[Text, Text] | None:
    if not state.instances:
        return None
    hourly = sum(
        (i.offer.spot_price if i.spot else i.offer.on_demand_price) or 0.0
        for i in state.instances
    )
    if hourly <= 0:
        return None
    elapsed_h = (time.monotonic() - state.pool_started_at) / 3600 if state.pool_started_at else 0
    total = hourly * elapsed_h
    rate = Text()
    rate.append(f" ${hourly:.2f}/hr ", style=_badge_style("cost"))
    cumul = Text()
    cumul.append(f" \u03a3 ${total:.2f} ", style=_badge_style("cost"))
    return rate, cumul


def _gauge_inline(label: str, pct: float) -> Text:
    t = Text()
    t.append(f" {label} {pct:.0f}% ", style=_gauge_badge(label, pct))
    return t


# --- Badge builders ---


def _styled_badge(label: str, style_key: str) -> Text:
    t = Text()
    t.append(f" {label} ", style=_badge_style(style_key))
    return t


def _workers_badge(ready: int, desired: int) -> Text:
    label = f"workers {ready}/{desired}"
    if ready >= desired:
        style = _make_badge(120, 0.50)
    elif ready == 0:
        style = _make_badge(0, 0.60)
    else:
        style = _make_badge(45, 0.45)
    t = Text()
    t.append(f" {label} ", style=style)
    return t


def _collect_badges(state: _State) -> tuple[list[Text], list[Text], list[Text]]:
    infra: list[Text] = []
    status: list[Text] = []
    tasks: list[Text] = []

    # --- Line 1: Infra ---
    infra.append(_inline_badge("skyward"))

    if state.instances:
        first = state.instances[0]
        itype = first.offer.instance_type
        n = len(state.instances)

        n_spot = sum(1 for i in state.instances if i.spot)
        n_od = n - n_spot

        infra.append(_styled_badge(f"{n}\u00d7", "cluster"))
        if n_spot:
            infra.append(_inline_badge("spot"))
        if n_od:
            infra.append(_inline_badge("on-demand"))
        if itype.name:
            infra.append(_inline_badge(itype.name))

        region = first.region or ""
        provider = (state.cluster.spec.provider or "").upper() if state.cluster else ""
        if region:
            infra.append(_inline_badge(region))
        if provider:
            infra.append(_inline_badge(f"☁️ {provider}"))

        vcpus = int(itype.vcpus * n)
        mem_gb = int(itype.memory_gb * n)
        if vcpus:
            infra.append(_inline_badge(f"{vcpus} vCPU"))
        if mem_gb:
            infra.append(_inline_badge(f"{mem_gb} GB"))

        accel = itype.accelerator
        if accel:
            total = accel.count * n
            mem = accel.memory or state.spec_accelerator_memory
            if not mem:
                vram_vals = _collect_metric_vals(state, "gpu_mem_total_mb")
                if vram_vals:
                    per_gpu = sum(vram_vals) / len(vram_vals)
                    mem = f"{per_gpu / 1024:.0f}GB" if per_gpu >= 1024 else f"{per_gpu:.0f}MB"
            mem_str = f" {mem}" if mem else ""
            infra.append(_inline_badge(f"\u26a1 {total}\u00d7 {accel.name}{mem_str}"))

    # --- Line 2: Status (phase + workers + metrics + reconciler + cost) ---
    match state.phase:
        case _Phase.PROVISIONING:
            status.append(_inline_badge("provisioning"))
        case _Phase.SSH:
            count = sum(1 for s in state.nodes.values() if s.value >= _NodeStatus.SSH.value)
            status.append(_inline_badge("connecting"))
            status.append(_styled_badge(f"ssh {count}/{state.total_nodes}", "connecting"))
        case _Phase.BOOTSTRAP | _Phase.WORKERS:
            phase_name = "bootstrap" if state.phase == _Phase.BOOTSTRAP else "workers"
            target = (
                _NodeStatus.BOOTSTRAPPING if state.phase == _Phase.BOOTSTRAP
                else _NodeStatus.READY
            )
            count = sum(1 for s in state.nodes.values() if s.value >= target.value)
            status.append(_inline_badge(phase_name))
            status.append(_styled_badge(f"{count}/{state.total_nodes} ready", phase_name))
        case _Phase.READY:
            status.append(_inline_badge("ready"))
            ready_count = sum(1 for s in state.nodes.values() if s == _NodeStatus.READY)
            desired = state.desired_nodes or state.total_nodes
            status.append(_workers_badge(ready_count, desired))

            # Metrics
            cpu_vals = _collect_metric_vals(state, "cpu")
            mem_vals = _collect_metric_vals(state, "mem")
            gpu_vals = _collect_metric_vals(state, "gpu_util")
            gpu_mem_vals = _collect_metric_vals(state, "gpu_mem_mb")
            gpu_mem_total = _collect_metric_vals(state, "gpu_mem_total_mb")
            if cpu_vals:
                status.append(_gauge_inline("cpu", sum(cpu_vals) / len(cpu_vals)))
            if mem_vals:
                status.append(_gauge_inline("mem", sum(mem_vals) / len(mem_vals)))
            if gpu_vals:
                status.append(_gauge_inline("gpu", sum(gpu_vals) / len(gpu_vals)))
            if gpu_mem_vals and gpu_mem_total:
                avg_used = sum(gpu_mem_vals) / len(gpu_mem_vals)
                avg_total = sum(gpu_mem_total) / len(gpu_mem_total)
                pct = (avg_used / avg_total * 100) if avg_total > 0 else 0
                status.append(_gauge_inline("vram", pct))

            # Reconciler
            match state.reconciler_state:
                case "scaling_up":
                    status.append(_styled_badge(
                        f"\u25cf scaling \u2192 {state.desired_nodes}", "scaling",
                    ))
                    if state.pending_nodes:
                        status.append(_styled_badge(f"pending {state.pending_nodes}", "scaling"))
                case "draining":
                    status.append(_styled_badge(
                        f"\u25cf draining {state.draining_nodes}", "drifted",
                    ))
                case _:
                    status.append(_inline_badge("in sync"))
            if state.is_elastic:
                if state.min_nodes is not None:
                    status.append(_inline_badge(f"min {state.min_nodes}"))
                status.append(_styled_badge(f"cur {len(state.nodes)}", "cluster"))
                if state.max_nodes is not None:
                    status.append(_inline_badge(f"max {state.max_nodes}"))

            # Cost
            cost = _cost_badges(state)
            if cost:
                status.extend(cost)

            # --- Line 3: Tasks ---
            if state.tasks_queued:
                tasks.append(_styled_badge(f"{state.tasks_queued} queued", "queued"))
            tasks.append(_styled_badge(f"\u25cf {state.tasks_running} running", "running"))
            tasks.append(_styled_badge(f"\u2714 {state.tasks_done} done", "done"))
            if state.tasks_failed:
                tasks.append(_styled_badge(f"\u2717 {state.tasks_failed} failed", "failed"))
            rate = _throughput(state)
            if rate > 0:
                tasks.append(_styled_badge(f"{rate:.1f} tasks/min", "running"))
                remaining = state.tasks_queued + state.tasks_running
                if remaining > 0:
                    eta_min = remaining / rate
                    tasks.append(_styled_badge(f"est. {_format_duration(eta_min * 60)}", "cost"))

        case _Phase.STOPPING:
            status.append(_inline_badge("shutting down"))
            if state.tasks_done:
                status.append(_styled_badge(f"\u2714 {state.tasks_done} done", "done"))
            rate = _throughput(state)
            if rate > 0:
                status.append(_styled_badge(f"{rate:.1f}/min", "running"))
            cost = _cost_badges(state)
            if cost:
                status.extend(cost)

    return infra, status, tasks


# --- Footer renderer ---


def _join_badges(badges: list[Text]) -> Text:
    combined = Text()
    for badge in badges:
        combined.append_text(badge)
    return combined


def _render_footer(state: _State) -> RenderableType:
    infra, status, tasks = _collect_badges(state)
    if not infra and not status and not tasks:
        return Text()

    lines: list[Align] = []
    for group in (infra, status, tasks):
        if group:
            lines.append(Align.center(_join_badges(group)))

    return Group(Text(), *lines)


# --- Session summary ---


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs:02d}s" if minutes < 60 else f"{minutes // 60}h {minutes % 60:02d}m"


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
        provider = (state.cluster.spec.provider or "").upper()
        region = state.instances[0].region if state.instances else ""
        itype = state.instances[0].offer.instance_type.name if state.instances else ""
        parts = [p for p in (provider, region, itype) if p]
        if parts:
            overview.add_row("Provider", Text(" \u00b7 ".join(parts)))

    if state.instances:
        n_spot = sum(1 for i in state.instances if i.spot)
        n_od = len(state.instances) - n_spot
        alloc_parts = []
        if n_spot:
            alloc_parts.append(f"$ {n_spot} spot")
        if n_od:
            alloc_parts.append(f"$$$ {n_od} on-demand")
        alloc = f" ({', '.join(alloc_parts)})" if alloc_parts else ""
        overview.add_row("Cluster", Text(f"{len(state.instances)} nodes{alloc}"))

        accel = state.instances[0].offer.instance_type.accelerator
        if accel:
            total = accel.count * len(state.instances)
            mem = f" {accel.memory}" if accel.memory else ""
            overview.add_row("Accelerator", Text(f"{total}\u00d7 {accel.name}{mem}"))

    overview.add_row("Duration", Text(_format_duration(duration)))

    hourly = sum(
        (i.offer.spot_price if i.spot else i.offer.on_demand_price) or 0.0
        for i in state.instances
    ) if state.instances else 0.0
    if hourly > 0:
        total_cost = hourly * (duration / 3600)
        cost_text = Text()
        cost_text.append(f"${total_cost:.2f}", style="green")
        cost_text.append(f" (${hourly:.2f}/hr)", style="bright_black")
        overview.add_row("Cost", cost_text)

    tasks_text = Text()
    tasks_text.append(f"{state.tasks_done} completed", style="green bold")
    tasks_text.append(" \u00b7 ", style="color(240)")
    tasks_text.append(f"{state.tasks_failed} failed", style="red bold")
    overview.add_row("Tasks", tasks_text)

    rate = _throughput(state, now=now)
    if rate > 0:
        overview.add_row("Throughput", Text(f"{rate:.1f} tasks/min"))

    if state.task_latencies:
        avg_lat = sum(state.task_latencies) / len(state.task_latencies)
        lo = min(state.task_latencies)
        hi = max(state.task_latencies)
        latency_text = Text()
        latency_text.append(f"{avg_lat:.1f}s")
        latency_text.append(f" (min {lo:.1f}s, max {hi:.1f}s)", style="bright_black")
        overview.add_row("Avg latency", latency_text)

    gpu_vals = _collect_metric_vals(state, "gpu_util")
    gpu_mem_vals = _collect_metric_vals(state, "gpu_mem_mb")
    gpu_mem_total = _collect_metric_vals(state, "gpu_mem_total_mb")
    cpu_vals = _collect_metric_vals(state, "cpu")
    mem_vals = _collect_metric_vals(state, "mem")

    if gpu_vals:
        avg_gpu = sum(gpu_vals) / len(gpu_vals)
        lo_g, hi_g = min(gpu_vals), max(gpu_vals)
        overview.add_row("Avg GPU", Text(f"{avg_gpu:.0f}% ({lo_g:.0f}%\u2013{hi_g:.0f}%)"))
    if gpu_mem_vals:
        avg_vram = sum(gpu_mem_vals) / len(gpu_mem_vals)
        if gpu_mem_total:
            total_vram = sum(gpu_mem_total) / len(gpu_mem_total)
            overview.add_row("Avg VRAM", Text(f"{avg_vram:.0f}/{total_vram:.0f} MB"))
        else:
            overview.add_row("Avg VRAM", Text(f"{avg_vram:.0f} MB"))
    if cpu_vals:
        avg_cpu = sum(cpu_vals) / len(cpu_vals)
        overview.add_row("Avg CPU", Text(f"{avg_cpu:.0f}%"))
    if mem_vals:
        avg_mem = sum(mem_vals) / len(mem_vals)
        overview.add_row("Avg Memory", Text(f"{avg_mem:.0f}%"))

    breakdown = Table(
        title="Task Execution Summary\n",
        title_style="bold",
        title_justify="center",
        show_edge=False,
        box=None,
        padding=(0, 2),
        header_style="bold bright_black",
    )
    breakdown.add_column("Task")
    breakdown.add_column("Calls", justify="right")
    breakdown.add_column("Avg", justify="right")
    breakdown.add_column("Min", justify="right")
    breakdown.add_column("Max", justify="right")
    breakdown.add_column("Failed", justify="right")

    all_fns = sorted(
        set(list(state.task_fn_stats.keys()) + list(state.task_fn_failed.keys())),
        key=lambda t: len(state.task_fn_stats.get(t, ())),
        reverse=True,
    )
    for fn_name in all_fns:
        latencies = state.task_fn_stats.get(fn_name, ())
        fails = state.task_fn_failed.get(fn_name, 0)
        calls = len(latencies) + fails
        avg = f"{sum(latencies) / len(latencies):.1f}s" if latencies else "\u2013"
        mn = f"{min(latencies):.1f}s" if latencies else "\u2013"
        mx = f"{max(latencies):.1f}s" if latencies else "\u2013"
        fail_text = Text(str(fails), style="red bold") if fails else Text("0", style="bright_black")
        breakdown.add_row(
            Text(fn_name), Text(str(calls)), Text(avg),
            Text(mn, style="green"), Text(mx, style="yellow"), fail_text,
        )

    if state.tasks_per_instance:
        counts = list(state.tasks_per_instance.values())
        lo, hi = min(counts), max(counts)
        avg_n = sum(counts) / len(counts)
        dist_text = Text()
        dist_text.append(f"avg {avg_n:.0f}")
        dist_text.append(f" (min {lo}, max {hi})", style="bright_black")
        overview.add_row("Distribution", dist_text)

    layout = Table.grid(padding=(0, 3), expand=True)
    layout.add_column("left", ratio=2)
    layout.add_column("right", ratio=3)
    layout.add_row(overview, breakdown)

    return Group(Text(""), layout, Text(""))


# =============================================================================
# Controller
# =============================================================================


@dataclass(frozen=True, slots=True)
class LocalOutput:
    line: str
    stream: str = "stdout"


type ConsoleInput = SpyEvent | LocalOutput


def _format_task(
    fn: object, args: tuple, kwargs: dict, max_sig: int = 80, max_arg: int = 12
) -> str:
    name = getattr(fn, "__name__", str(fn))
    parts = [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]

    if not parts:
        return name

    sig = ", ".join(parts)
    if len(sig) <= max_sig:
        return f"{name}({sig})"

    truncated = [p if len(p) <= max_arg else p[:max_arg] + "\u2026" for p in parts]
    included: list[str] = []
    length = 0

    for part in truncated:
        extra = len(", ") if included else 0
        if length + extra + len(part) + len(", \u2026") > max_sig:
            break
        length += extra + len(part)
        included.append(part)

    match included:
        case []:
            return f"{name}(\u2026)"
        case _ if len(included) < len(parts):
            return f"{name}({', '.join(included)}, \u2026)"
        case _:
            return f"{name}({', '.join(included)})"


def _node_id_from_path(actor_path: str) -> int | None:
    import re
    m = re.search(r"node-(\d+)", actor_path)
    return int(m.group(1)) if m else None


def _resolve_instance_id(state: _State, node_id: int | None = None) -> str | None:
    if node_id is not None and node_id < len(state.instances):
        return state.instances[node_id].id
    return None


def console_actor(spec: PoolSpec) -> Behavior[ConsoleInput]:
    """Console tells this story: idle -> observing -> stopped."""

    console = Console(stderr=True)
    _original_stdout: list[Any] = []

    def _install_stdout_redirect(ctx: ActorContext[ConsoleInput]) -> None:
        ref = ctx.self

        class _Writer:
            def __init__(self, original: Any) -> None:
                self._original = original

            def write(self, s: str) -> int:
                for line in s.splitlines(keepends=True):
                    if stripped := line.rstrip():
                        ref.tell(LocalOutput(line=stripped))
                return len(s)

            def flush(self) -> None:
                pass

            @property
            def encoding(self) -> str:
                return self._original.encoding

            @property
            def errors(self) -> str | None:
                return self._original.errors

            def fileno(self) -> int:
                return self._original.fileno()

            def isatty(self) -> bool:
                return False

        import sys

        _original_stdout.append(sys.stdout)
        sys.stdout = _Writer(sys.stdout)  # type: ignore[assignment]

    async def _restore_stdout(_ctx: ActorContext[ConsoleInput]) -> None:
        import sys

        if _original_stdout:
            sys.stdout = _original_stdout.pop()

    live: Live | None = None
    live_stopped = False

    def _update_footer(state: _State) -> None:
        nonlocal live
        if live_stopped:
            return
        renderable = _render_footer(state)
        if live is None:
            live = Live(
                renderable, console=console,
                refresh_per_second=8, screen=False,
                redirect_stdout=False, redirect_stderr=False,
            )
            live.start()
        else:
            live.update(renderable)

    def _stop_live() -> None:
        nonlocal live, live_stopped
        live_stopped = True
        if live is not None:
            live.stop()
            live = None

    def idle() -> Behavior[ConsoleInput]:
        async def setup(ctx: ActorContext[ConsoleInput]) -> Behavior[ConsoleInput]:
            _install_stdout_redirect(ctx)
            from skyward import __version__

            line1 = Text()
            line1.append(f" v{__version__} ", style=_make_badge(140, 0.6))
            line1.append("  Cloud accelerators with a single decorator", style=DIM)

            line2 = Text()
            line2.append("https://gabfssilva.github.io/skyward/", style="underline dim")

            right = [Text(), line1, line2, Text()]

            banner = Table.grid(padding=(0, 2))
            banner.add_column("logo")
            banner.add_column("info")
            for logo_line, info_line in zip(_LOGO_LINES, right, strict=True):
                banner.add_row(logo_line, info_line)
            console.print()
            console.print(banner)
            console.print()
            accel_mem = ""
            match spec.accelerator:
                case str(name):
                    import contextlib

                    from skyward.accelerators import Accelerator
                    with contextlib.suppress(ValueError):
                        accel_mem = Accelerator.from_name(name).memory
                case accel if accel is not None:
                    accel_mem = accel.memory
            state = _State(
                total_nodes=spec.nodes,
                desired_nodes=spec.nodes,
                min_nodes=spec.min_nodes,
                max_nodes=spec.max_nodes,
                is_elastic=spec.auto_scaling,
                spec_accelerator_memory=accel_mem,
            )
            behavior = observing(state)
            return Behaviors.with_lifecycle(behavior, post_stop=_restore_stdout)

        return Behaviors.setup(setup)

    def observing(state: _State) -> Behavior[ConsoleInput]:
        async def receive(
            ctx: ActorContext[ConsoleInput], msg: ConsoleInput,
        ) -> Behavior[ConsoleInput]:
            match msg:
                case SpyEvent(event=Terminated()):
                    return Behaviors.same()

                case SpyEvent(event=StartPool()):
                    new = _on_start_pool(state)
                    _update_footer(new)
                    return observing(new)

                case SpyEvent(event=ClusterReady()):
                    new = _on_cluster_ready(state)
                    _update_footer(new)
                    return observing(new)

                case SpyEvent(event=InstancesProvisioned(cluster=cluster, instances=raw)):
                    new = _on_instances_provisioned(state, cluster, tuple(raw))
                    _update_footer(new)
                    return observing(new)

                case SpyEvent(event=PoolStarted()):
                    _emit(console, "skyward", "\u2713 Pool ready", "green bold")
                    new = replace(state, phase=_Phase.READY)
                    _update_footer(new)
                    return observing(new)

                case SpyEvent(event=StopPool()):
                    _stop_live()
                    _emit(console, "skyward", "Shutting down...", "yellow")
                    summary = _render_summary(state)
                    console.print(summary)
                    return observing(replace(state, phase=_Phase.STOPPING))

                case SpyEvent(event=PoolStopped() | _ShutdownDone()):
                    return Behaviors.same()

                case SpyEvent(event=Provision() | NodeBecameReady() | _PollResult()):
                    return Behaviors.same()

                case SpyEvent(event=NodeLost(node_id=nid, reason=reason)):
                    _emit(console, "error", f"Node {nid} lost: {reason}", "red")
                    return Behaviors.same()

                case SpyEvent(event=DesiredCountChanged(desired=desired)):
                    new = _on_desired_changed(state, desired)
                    _update_footer(new)
                    return observing(new)

                case SpyEvent(event=SpawnNodes(instances=insts, cluster=cl)):
                    new = _on_spawn_nodes(state, tuple(insts), cl)
                    _update_footer(new)
                    return observing(new)

                case SpyEvent(event=NodeJoined()):
                    new = _on_node_joined(state)
                    _update_footer(new)
                    return observing(new)

                case SpyEvent(event=DrainNode()):
                    new = _on_drain_node(state)
                    _update_footer(new)
                    return observing(new)

                case SpyEvent(event=DrainComplete(instance_id=iid)):
                    new = _on_drain_complete(state, iid)
                    _update_footer(new)
                    return observing(new)

                case SpyEvent(event=ReconcilerNodeLost()):
                    new = _on_reconciler_node_lost(state)
                    _update_footer(new)
                    return observing(new)

                case SpyEvent(actor_path=path, event=_Connected()):
                    nid = _node_id_from_path(path)
                    iid = _resolve_instance_id(state, node_id=nid)
                    if iid:
                        _emit(console, iid, "\u2713 SSH connected", "green")
                        new = _on_ssh_connected(state, iid)
                        _update_footer(new)
                        return observing(new)
                    return Behaviors.same()

                case SpyEvent(event=_ConnectionFailed(error=error)):
                    _emit(console, "error", f"SSH failed: {error}", "red")
                    return Behaviors.same()

                case SpyEvent(event=Preempted(reason=reason)):
                    _emit(console, "error", f"Preempted: {reason}", "red")
                    return Behaviors.same()

                case SpyEvent(event=BootstrapConsole() as ev):
                    content = ev.content.strip()
                    if content and not content.startswith("#"):
                        _emit(console, ev.instance.instance.id, content[:120])
                    return Behaviors.same()

                case SpyEvent(event=BootstrapPhase() as ev):
                    iid = ev.instance.instance.id
                    match ev.event:
                        case "started":
                            _emit(console, iid, f"\u25b8 {ev.phase}...")
                        case "completed":
                            elapsed = f" ({ev.elapsed:.1f}s)" if ev.elapsed else ""
                            _emit(console, iid, f"\u2713 {ev.phase}{elapsed}", "green")
                        case "failed":
                            _emit(console, iid, f"\u2717 {ev.phase}: {ev.error}", "red")
                    return Behaviors.same()

                case SpyEvent(event=BootstrapCommand() as ev):
                    cmd = ev.command.strip()
                    if cmd:
                        display = f"$ {cmd[:80]}..." if len(cmd) > 80 else f"$ {cmd}"
                        _emit(console, ev.instance.instance.id, display, "dim")
                    return Behaviors.same()

                case SpyEvent(event=BootstrapDone(instance=inst, success=ok, error=err)):
                    iid = inst.instance.id
                    if ok:
                        new = _on_bootstrap_done(state, iid)
                        _update_footer(new)
                        return observing(new)
                    _emit(console, iid, f"\u2717 Bootstrap failed: {err}", "red")
                    return Behaviors.same()

                case SpyEvent(event=_LocalInstallDone() | _UserCodeSyncDone()):
                    return Behaviors.same()

                case SpyEvent(event=_PostBootstrapFailed(error=err)):
                    _emit(console, "error", f"Post-bootstrap failed: {err}", "red")
                    return Behaviors.same()

                case SpyEvent(actor_path=path, event=_WorkerStarted()):
                    nid = _node_id_from_path(path)
                    iid = _resolve_instance_id(state, node_id=nid)
                    if iid:
                        _emit(console, iid, "\u2713 Worker joined", "green")
                        new = _on_worker_started(state, iid)
                        _update_footer(new)
                        return observing(new)
                    return Behaviors.same()

                case SpyEvent(event=_WorkerFailed(error=error)):
                    _emit(console, "error", f"Worker failed: {error}", "red")
                    return Behaviors.same()

                case SpyEvent(event=SubmitTask(task_id=tid) as ev) if tid not in state.inflight:
                    name = _format_task(ev.fn, ev.args, ev.kwargs)
                    _emit_task(console, "skyward", "queued", name)
                    new = _on_task_submitted(state, tid, name, "single")
                    _update_footer(new)
                    return observing(new)

                case SpyEvent(
                    event=SubmitBroadcast(task_id=tid) as ev,
                ) if tid not in state.inflight:
                    name = _format_task(ev.fn, ev.args, ev.kwargs)
                    n = len(state.instances)
                    _emit_task(console, "skyward", "queued", f"{name} \u2192 all {n} nodes")
                    new = _on_task_submitted(state, tid, name, "broadcast")
                    _update_footer(new)
                    return observing(new)

                case SpyEvent(event=SubmitTask() | SubmitBroadcast()):
                    return Behaviors.same()

                case SpyEvent(event=TaskSubmitted(task_id=tid, node_id=nid)):
                    iid = _resolve_instance_id(state, node_id=nid) or ""
                    if iid:
                        entry = state.inflight.get(tid)
                        if entry:
                            _emit_task(console, iid, "running", entry.name)
                    new = _on_task_assigned(state, tid, iid)
                    return observing(new)

                case SpyEvent(event=TaskResult(task_id=tid, node_id=nid, error=is_err)):
                    entry = state.inflight.get(tid)
                    if entry is None:
                        return Behaviors.same()
                    iid = entry.instance_id or _resolve_instance_id(state, node_id=nid) or "skyward"
                    match entry.kind:
                        case "broadcast":
                            new = _on_broadcast_partial(state, tid)
                            updated = new.inflight.get(tid)
                            if updated and updated.broadcast_done >= updated.broadcast_total:
                                elapsed = time.monotonic() - entry.started_at
                                _emit_task(console, iid, "done", f"{entry.name} in {elapsed:.1f}s")
                                new = _on_task_done(new, tid, elapsed)
                            _update_footer(new)
                            return observing(new)
                        case _:
                            elapsed = time.monotonic() - entry.started_at
                            if is_err:
                                _emit_task(console, iid, "failed", entry.name)
                                new = _on_task_failed(state, tid)
                            else:
                                _emit_task(console, iid, "done", f"{entry.name} in {elapsed:.1f}s")
                                new = _on_task_done(state, tid, elapsed)
                            _update_footer(new)
                            return observing(new)

                case SpyEvent(event=Log() as ev):
                    line = ev.line.strip()
                    if line:
                        _emit(console, ev.instance.instance.id, line[:120])
                    return Behaviors.same()

                case SpyEvent(event=Metric() as ev):
                    new = _on_metric(state, ev.instance.instance.id, ev.name, ev.value)
                    _update_footer(new)
                    return observing(new)

                case SpyEvent(event=ExecuteOnNode()):
                    return Behaviors.same()

                case SpyEvent(event=Error() as ev):
                    style = "red bold" if ev.fatal else "red"
                    _emit(console, "error", ev.message, style)
                    return Behaviors.same()

                case SpyEvent(event=ShutdownRequested()):
                    _stop_live()
                    _emit(console, "skyward", "Shutting down...", "yellow")
                    summary = _render_summary(state)
                    console.print(summary)
                    return observing(replace(state, phase=_Phase.STOPPING))

                case LocalOutput(line=line):
                    if stripped := line.rstrip():
                        _emit(console, "local", stripped)
                    return Behaviors.same()

                case _:
                    return Behaviors.same()

        return Behaviors.receive(receive)

    return idle()
