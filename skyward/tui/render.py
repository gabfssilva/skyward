"""Rich renderables for the TUI.

Pure functions: ``(data, palette, ...) -> RenderableType``.  Widgets stay
thin — they hold no rendering logic, only call these and push the result
into a ``Static`` (via ``update``) or a ``RichLog`` (via ``write``).
"""

from __future__ import annotations

from collections.abc import Sequence

from rich.console import Group, RenderableType
from rich.style import Style
from rich.table import Table
from rich.text import Text

from skyward.tui.model import UiCluster, UiLog, UiNode, UiStatus, clock, uptime
from skyward.tui.theme import NODE_COLORS, PULSING, STATUS, Palette

__all__ = [
    "breadcrumb",
    "log_rows",
    "log_text",
    "metrics",
    "node_badge",
    "phase_bar",
    "summary",
    "topbar",
    "tree",
]

_LEVEL = {"INFO": "dim", "OK": "green", "WARN": "amber", "ERR": "red"}


def _node_color(node_id: str) -> str:
    try:
        idx = int(node_id.split("-")[1])
    except (IndexError, ValueError):
        idx = 0
    return NODE_COLORS[idx % len(NODE_COLORS)]


def _blend(fg: str, bg: str, alpha: float) -> str:
    """Return ``fg`` composited over ``bg`` at ``alpha`` (terminals lack alpha)."""
    fr, fgn, fb = int(fg[1:3], 16), int(fg[3:5], 16), int(fg[5:7], 16)
    br, bgn, bb = int(bg[1:3], 16), int(bg[3:5], 16), int(bg[5:7], 16)
    r = round(fr * alpha + br * (1 - alpha))
    g = round(fgn * alpha + bgn * (1 - alpha))
    b = round(fb * alpha + bb * (1 - alpha))
    return f"#{r:02x}{g:02x}{b:02x}"


def node_badge(node_id: str, pal: Palette) -> Text:
    """Return a colour-keyed pill ``Text`` for a node id (tinted background)."""
    color = _node_color(node_id)
    return Text(
        f" {node_id} ",
        Style(color=color, bgcolor=_blend(color, pal.panel, 0.22), bold=True),
    )


def _dot(status: UiStatus, pal: Palette, pulse_on: bool) -> Text:
    attr, glyph, _ = STATUS[status]
    dim = status in PULSING and not pulse_on
    return Text(glyph, Style(color=getattr(pal, attr), dim=dim))


def _bar(value: float, maximum: float, color: str, pal: Palette, width: int = 20) -> Text:
    filled = 0 if not maximum else max(0, min(width, round(value / maximum * width)))
    t = Text()
    t.append("█" * filled, Style(color=color))
    t.append("░" * (width - filled), Style(color=pal.faint))
    return t


def _gauge_color(ratio: float, pal: Palette) -> str:
    if ratio > 0.85:
        return pal.red
    if ratio > 0.65:
        return pal.amber
    return pal.green


def _log_row(log: UiLog, pal: Palette, *, show_node: bool) -> Text:
    t = Text()
    t.append(f"{clock(log.ts):>8}  ", Style(color=pal.faint))
    if show_node and log.node:
        badge = node_badge(log.node, pal)
        badge.pad_right(max(0, 8 - badge.cell_len))
        t.append_text(badge)
        t.append(" ")
    t.append(f"{log.level:<4} ", Style(color=getattr(pal, _LEVEL.get(log.level, "dim"))))
    t.append(log.msg, Style(color=pal.text))
    return t


def log_rows(logs: Sequence[UiLog], pal: Palette, *, show_node: bool) -> list[Text]:
    """Render a slice of log entries into styled rows."""
    return [_log_row(log, pal, show_node=show_node) for log in logs]


def log_text(logs: Sequence[UiLog], *, show_node: bool) -> str:
    """Render log entries as plain text for the clipboard."""
    lines: list[str] = []
    for log in logs:
        prefix = f"{log.node:<8} " if show_node and log.node else ""
        lines.append(f"{clock(log.ts):>8}  {prefix}{log.level:<4} {log.msg}")
    return "\n".join(lines)


def topbar(cluster: UiCluster, pal: Palette) -> RenderableType:
    """Top bar: logo · pool name … ready/total · clock."""
    left = Text()
    left.append("▲ skyward", Style(color=pal.accent, bold=True))
    left.append("   ·   ", Style(color=pal.faint))
    left.append(cluster.name, Style(color=pal.text))
    right = Text(
        f"{cluster.ready_count}/{len(cluster.nodes)} ready · {clock(cluster.elapsed)}",
        Style(color=pal.dim),
    )
    grid = Table.grid(expand=True)
    grid.add_column(ratio=1)
    grid.add_column(justify="right")
    grid.add_row(left, right)
    return grid


def summary(cluster: UiCluster, pal: Palette) -> RenderableType:
    """Two-line dashboard summary header."""
    ready = [n for n in cluster.nodes if n.status is UiStatus.READY]
    n = len(cluster.nodes)
    avg_gpu = round(sum(x.gpu for x in ready) / len(ready)) if ready else 0
    mem_used = sum(x.mem for x in ready)
    avg_temp = round(sum(x.temp for x in ready) / len(ready)) if ready else 0

    line1 = Text()
    line1.append(cluster.provider, Style(color=pal.text))
    line1.append(f" · {cluster.accel} ×{n} · {cluster.region} · ", Style(color=pal.dim))
    line1.append(cluster.allocation, Style(color=pal.text))

    line2 = Text(style=Style(color=pal.dim))
    line2.append(f"{cluster.ready_count}/{n} ready", Style(color=pal.text))
    line2.append(
        f" · gpu {avg_gpu}% · mem {mem_used:.0f}/{80 * n}G · {avg_temp}°C avg"
        f" · ${cluster.cost:.3f} · ${cluster.hourly_cost:.2f}/hr"
    )
    return Group(line1, line2)


def phase_bar(node: UiNode, pal: Palette, pulse_on: bool) -> RenderableType:
    """Bootstrap phase progress: ``✓ done → ◐ active → · pending``."""
    t = Text()
    for i, phase in enumerate(node.phases):
        if i:
            t.append("  →  ", Style(color=pal.faint))
        if i < node.phase_idx:
            t.append(f"✓ {phase}", Style(color=pal.green))
        elif i == node.phase_idx:
            t.append(f"◐ {phase}", Style(color=pal.accent, dim=not pulse_on))
        else:
            t.append(f"· {phase}", Style(color=pal.dim))
    return t


def metrics(node: UiNode, pal: Palette, pulse_on: bool) -> RenderableType:
    """Per-node metrics panel body."""
    if node.status is not UiStatus.READY:
        _, _, label = STATUS[node.status]
        head = Text.assemble(_dot(node.status, pal, pulse_on), (f" {label} — gpu offline", Style(color=pal.dim)))
        note = Text("metrics resume when the node rejoins the cluster.", Style(color=pal.dim))
        return Group(head, Text(), note)

    grid = Table.grid(padding=(0, 2))
    grid.add_column(width=8)
    grid.add_column()
    grid.add_column(justify="right")
    rows = (
        ("gpu util", _bar(node.gpu, 100, _gauge_color(node.gpu / 100, pal), pal), f"{node.gpu:.0f} %"),
        ("gpu mem", _bar(node.mem, 80, pal.accent, pal), f"{node.mem:.1f} / 80 GB"),
        ("gpu temp", _bar(node.temp, 95, _gauge_color(node.temp / 95, pal), pal), f"{node.temp:.0f} °C"),
        ("cpu", _bar(34, 100, pal.green, pal), "34 %"),
        ("disk", _bar(41, 100, pal.green, pal), "41 % /data"),
    )
    for label, bar, value in rows:
        grid.add_row(Text(label, Style(color=pal.dim)), bar, Text(value, Style(color=pal.text)))

    link = Text()
    link.append("nvlink p2p ", Style(color=pal.dim))
    link.append("47.2 GB/s", Style(color=pal.text))
    link.append(" · pcie ", Style(color=pal.dim))
    link.append("12.1 GB/s", Style(color=pal.text))
    return Group(grid, Text(), link)


def breadcrumb(cluster: UiCluster, node: UiNode, pal: Palette, pulse_on: bool) -> RenderableType:
    """Node-view breadcrumb header."""
    attr, _, label = STATUS[node.status]
    line1 = Text()
    line1.append("dashboard", Style(color=pal.dim))
    line1.append(" / ", Style(color=pal.faint))
    line1.append(node.id, Style(color=pal.accent))
    line1.append(" · ", Style(color=pal.dim))
    line1.append_text(_dot(node.status, pal, pulse_on))
    line1.append(" ")
    line1.append(label, Style(color=getattr(pal, attr)))

    rank = "0 (coordinator)" if node.role == "coord" else str(node.rank)
    line2 = Text(
        f"rank {rank} · {cluster.provider} · {cluster.accel} · {node.ip} · up {uptime(node, cluster.t)}",
        Style(color=pal.dim),
    )
    return Group(line1, line2)


def tree(
    cluster: UiCluster,
    pal: Palette,
    *,
    selected: int,
    open_nodes: frozenset[int],
    root_open: bool,
    pulse_on: bool,
) -> RenderableType:
    """Collapsible node tree with full-width selection highlight."""
    grid = Table.grid(expand=True, padding=0)
    grid.add_column(ratio=1)
    grid.add_column(justify="right")

    root_left = Text()
    root_left.append(("▾ " if root_open else "▸ "), Style(color=pal.dim))
    root_left.append(cluster.name, Style(color=pal.text))
    grid.add_row(root_left, Text(f"{cluster.ready_count}/{len(cluster.nodes)}", Style(color=pal.dim)))

    if not root_open:
        return grid

    nodes = cluster.nodes
    for i, node in enumerate(nodes):
        last = i == len(nodes) - 1
        is_open = i in open_nodes
        attr, _, status_label = STATUS[node.status]
        ready = node.status is UiStatus.READY

        left = Text()
        left.append(("└─" if last else "├─"), Style(color=pal.faint))
        left.append((" ▾ " if is_open else " ▸ "), Style(color=pal.dim))
        left.append_text(_dot(node.status, pal, pulse_on))
        left.append(" ")
        left.append(node.id, Style(color=pal.accent if i == selected else pal.text))

        right = (
            Text(f"{node.gpu:.0f}%", Style(color=pal.text))
            if ready
            else Text(status_label, Style(color=getattr(pal, attr)))
        )
        row_style = Style(bgcolor=pal.tint) if i == selected else None
        grid.add_row(left, right, style=row_style)

        if not is_open:
            continue
        prefix = "  " if last else "│ "
        leaves = (
            [
                f"rank {node.rank} · {node.role}",
                node.ip,
                f"gpu {node.gpu:.0f}%  mem {node.mem:.0f}G  {node.temp:.0f}°C",
                f"up {uptime(node, cluster.t)}",
            ]
            if ready
            else [f"rank {node.rank} · {node.role}", node.ip, f"{status_label} — gpu offline"]
        )
        for j, leaf in enumerate(leaves):
            guide = prefix + ("└─" if j == len(leaves) - 1 else "├─")
            cell = Text()
            cell.append(guide + " ", Style(color=pal.faint))
            cell.append(leaf, Style(color=pal.dim))
            grid.add_row(cell, Text(""))

    return grid
