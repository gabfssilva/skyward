from __future__ import annotations

import time
from types import MappingProxyType

from rich.align import Align
from rich.console import Console, Group, RenderableType
from rich.style import Style
from rich.table import Table
from rich.text import Text

from .model import _throughput
from .state import _NodeStatus, _Phase, _State

_LOGO_LINES = (
"   \u258c           \u258c",
" \u259b\u2598\u2599\u2598\u258c\u258c\u258c\u258c\u258c\u2580\u258c\u259b\u2598\u259b\u258c",
" \u2584\u258c\u259b\u2596\u2599\u258c\u259a\u259a\u2598\u2588\u258c\u258c \u2599\u258c",
"     \u2584\u258c")

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
WARNING_STYLE = "yellow" if _DARK else "dark_orange"


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


def _badge_text(label: str, link: str = "") -> Text:
    short = label[:8].center(8) if len(label) > 8 else label.center(8)
    t = Text()
    style = _badge_style(label)
    if link:
        style = style + Style(link=link)
    t.append(f" {short} ", style=style)
    return t


def _ssh_url(state: _State, instance_id: str) -> str:
    inst = next((i for i in state.instances if i.id == instance_id), None)
    if not inst or not inst.ip or not state.ssh_user:
        return ""
    port = f":{inst.ssh_port}" if inst.ssh_port != 22 else ""
    return f"ssh://{state.ssh_user}@{inst.ip}{port}"


def _ssh_command(state: _State, instance_id: str) -> str:
    inst = next((i for i in state.instances if i.id == instance_id), None)
    if not inst or not inst.ip or not state.ssh_user:
        return ""
    parts = ["ssh"]
    if state.ssh_key_path:
        parts.append(f'-i "{state.ssh_key_path}"')
    if inst.ssh_port != 22:
        parts.append(f"-p {inst.ssh_port}")
    parts.append(f"{state.ssh_user}@{inst.ip}")
    return " ".join(parts)


def _inline_badge(label: str) -> Text:
    t = Text()
    t.append(f" {label} ", style=_badge_style(label))
    return t


def _emit(console: Console, badge: str, text: str, style: str = "", link: str = "") -> None:
    line = _badge_text(badge, link=link)
    text_style = Style.parse(style) + Style(link=link) if link else (style or None)
    line.append(f"  {text}", style=text_style)
    console.print(line)


def _emit_task(console: Console, badge: str, status: str, text: str, link: str = "") -> None:
    line = _badge_text(badge, link=link)
    line.append(" ")
    line.append_text(_inline_badge(status))
    text_style = Style(link=link) if link else None
    line.append(f" {text}", style=text_style)
    console.print(line)


# --- Metric helpers ---


def _find_metrics(
    raw: MappingProxyType[str, float], *prefixes: str,
) -> list[float]:
    """Return all metric values matching any prefix.

    Exact match takes priority — if ``"mem"`` is a key, only that value is
    returned (avoids mixing the percentage with ``mem_used_mb`` /
    ``mem_total_mb``).  The ``prefix_*`` wildcard search only runs when
    there is no exact key, which is the case for multi-GPU / MIG metrics
    (e.g. ``gpu_util_0``, ``gpu_util_1``).
    """
    vals: list[float] = []
    for prefix in prefixes:
        if prefix in raw:
            vals.append(raw[prefix])
        else:
            for key, val in raw.items():
                if key.startswith(f"{prefix}_"):
                    vals.append(val)
    return vals


def _collect_metric_vals(
    state: _State, *prefixes: str,
) -> list[float]:
    vals: list[float] = []
    for iid in state.metrics:
        vals.extend(_find_metrics(state.metrics[iid], *prefixes))
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
            infra.append(_inline_badge(f"\u2601\ufe0f {provider}"))

        vcpus = int(itype.vcpus * n)
        mem_gb = int(itype.memory_gb * n)
        if vcpus:
            infra.append(_inline_badge(f"{vcpus} vCPU"))
        if mem_gb:
            infra.append(_inline_badge(f"{mem_gb} GB"))

        accel = itype.accelerator
        if accel:
            total = accel.count * n
            total_str = str(int(total)) if total == int(total) else f"{total:.1f}"
            mem = accel.memory or state.spec_accelerator_memory
            if not mem:
                vram_vals = _collect_metric_vals(state, "gpu_mem_total_mb")
                if vram_vals:
                    per_gpu = sum(vram_vals) / len(vram_vals)
                    mem = f"{per_gpu / 1024:.0f}GB" if per_gpu >= 1024 else f"{per_gpu:.0f}MB"
            mem_str = f" {mem}" if mem else ""
            infra.append(_inline_badge(f"\u26a1 {total_str}\u00d7 {accel.name}{mem_str}"))

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
            total_str = str(int(total)) if total == int(total) else f"{total:.1f}"
            mem = f" {accel.memory}" if accel.memory else ""
            overview.add_row("Accelerator", Text(f"{total_str}\u00d7 {accel.name}{mem}"))

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
            Text(mn, style="green"), Text(mx, style=WARNING_STYLE), fail_text,
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
