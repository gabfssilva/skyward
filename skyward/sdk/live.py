"""The pool as a picture, for a terminal that can hold one.

The line console says one thing at a time and forgets it; this keeps a picture
of the whole pool pinned to the bottom of the screen while the log scrolls above
it. Same events, same stream — what changes is that a state is now a place on the
screen that updates, not a line that goes by.

The picture has one grammar at every scale: the fleet is a honeycomb of numbered
hexagons — colour is the node's state, brightness is its GPU — and the cells only
change size with the pool. Two nodes are two huge hexes; two hundred fall back to
flat blocks. A cell never carries text beyond its rank: everything else lives in the
inspector, a fixed panel beside the wall showing the node under the cursor, with
the same sections in the same places at every zoom. The arrows walk the wall in
two dimensions, ``a`` jumps to the next node that needs attention, ``/`` goes to
a rank, ``q`` leaves. There are no screens, no tabs, no pagination.

The log scrolls above the panel only once its node is ready: a bootstrap speaks
through the inspector, not through four hundred lines of apt. What it said is
not lost — it is the tail in the inspector's output section.

Everything renders to stderr, like the line console: a pool that drew its panel
into a script's stdout would corrupt every pipeline it was ever put in.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import select
import sys
import termios
import time
import tty
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from contextlib import aclosing, contextmanager, suppress
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Literal, TextIO

import msgspec
from rich.console import Console as RichConsole
from rich.console import Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.style import Style
from rich.table import Table
from rich.text import Text

from skyward import plugins
from skyward.protocol.schemas import Compute, Function, Image, Node, Page, Task
from skyward.sdk.client import Client
from skyward.sdk.console import render as log_line

POLL = 2.0
"""How often the structural half is re-read. The events carry the rest, faster."""

INSPECTOR = 40
"""Columns the inspector panel keeps for itself, border included."""

TAIL = 40
"""Console lines kept per node, for the tail the inspector shows."""

OUTPUT = 40
"""Output lines kept per task, folded into its node's story."""

HISTORY = 12
"""Readings kept per metric, for the sparklines the inspector draws."""

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

type Key = Literal["up", "down", "left", "right", "enter", "escape", "backspace", "tab"] | tuple[Literal["char"], str]

_STATE_STYLE = {
    "requested": "grey50",
    "provisioning": "yellow",
    "connecting": "cyan",
    "bootstrapping": "blue",
    "ready": "green",
    "draining": "yellow",
    "lost": "red",
    "deleting": "grey50",
    "deleted": "grey50",
    "failed": "red",
    "degraded": "red",
}
"""A lifecycle word's colour — by what it means, matching the line console."""

_STAGE = {
    "requested": "provision",
    "provisioning": "provision",
    "connecting": "connect",
    "bootstrapping": "bootstrap",
    "ready": "ready",
    "draining": "drain",
    "deleting": "gone",
    "deleted": "gone",
    "lost": "lost",
    "failed": "failed",
    "degraded": "degraded",
}
"""The journey word a state belongs to: what the wall and the inspector both speak."""

_PRE_READY = frozenset({"requested", "provisioning", "connecting", "bootstrapping"})

_BROKEN = frozenset({"lost", "failed", "degraded"})

_WAITING = {
    "requested": "waiting for a machine",
    "provisioning": "waiting for the machine",
    "connecting": "waiting for ssh",
}
"""What a pre-ready node is waiting on, for the stretch before it has said anything."""

_METRICS = ("gpu_util", "gpu_mem_mb", "gpu_mem_total_mb", "cpu", "mem_used_mb", "mem_total_mb")
"""The six readings the machine collects; the four shown are derived from them."""

_SCROLLED = frozenset({
    "node.console",
    "node.failed",
    "node.lost",
    "compute.ready",
    "compute.degraded",
    "compute.failed",
    "compute.deleted",
})
"""What may earn a permanent line above the panel: the output, and the turns worth keeping.

The mundane lifecycle — requested, connecting, bootstrapping — is the panel's whole
job. Scrolling it as well would say everything twice, once moving and once still.
A node's console additionally waits for its node to be ready: the bootstrap talks
through the inspector, and its chatter goes to the node's tail instead.
"""


class _Reading(msgspec.Struct, frozen=True):
    """One `node.metrics` event, off the wire."""

    node: str
    name: str
    value: float


class _Accrued(msgspec.Struct, frozen=True):
    """One `compute.cost` gauge, off the wire."""

    cost: float


@dataclass(frozen=True, slots=True)
class PhaseMark:
    """One bootstrap phase, as far as the node has got with it."""

    name: str
    started: datetime | None = None
    finished: datetime | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class NodeRow:
    """One machine, as the panel knows it: structure from the view, the rest from the stream."""

    id: str
    state: str = ""
    rank: int = 0
    machine: str | None = None
    address: str | None = None
    accelerator: str | None = None
    price: float | None = None
    region: str | None = None
    error: str | None = None
    created_at: datetime | None = None
    state_at: Mapping[str, datetime] = field(default_factory=lambda: MappingProxyType({}))
    metrics: Mapping[str, tuple[float, ...]] = field(default_factory=lambda: MappingProxyType({}))
    phases: tuple[PhaseMark, ...] = ()
    tail: tuple[str, ...] = ()
    installed: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    fetching: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True, slots=True)
class TaskRow:
    """One task, as the panel knows it: what runs, where, and how it is going."""

    id: str
    state: str = "queued"
    function: str = ""
    node: str | None = None
    submitted_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    retries: int = 0
    error: str | None = None


@dataclass(frozen=True, slots=True)
class Pool:
    """The whole pool in one line: what it is, where, and how much of it is up.

    The state here is the compute's own — what the reconciler thinks of the pool —
    and never derived from the node lines below it. A pool that has been ready for
    ten days stays ``ready`` while a replacement node boots underneath.
    """

    name: str = ""
    provider: str = ""
    region: str | None = None
    accelerator: str | None = None
    allocation: str = ""
    state: str = "requested"
    ready: int = 0
    total: int = 0
    cost: float | None = None


@dataclass(frozen=True, slots=True)
class Ui:
    """Where the keyboard has put the viewer: which block the cursor is on.

    This is the whole difference a keypress makes. It rides in the ``View`` next to
    the data so that a command is a pure function of the two — nothing about the
    terminal leaks in. ``find`` is the rank being typed after ``/``.
    """

    cursor: int = 0
    find: str | None = None
    quitting: bool = False


@dataclass(frozen=True, slots=True)
class View:
    """Everything the pinned panel draws. The log is not here — it scrolled past."""

    pool: Pool = field(default_factory=Pool)
    nodes: tuple[NodeRow, ...] = ()
    plan: tuple[str, ...] = ()
    steps: tuple[str, ...] = ()
    tasks: tuple[TaskRow, ...] = ()
    outputs: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: MappingProxyType({}))
    ui: Ui = field(default_factory=Ui)


# --------------------------------------------------------------------------- #
# reducers: fold the fast stream, merge the slow poll, apply the keyboard      #
# --------------------------------------------------------------------------- #


def observe(view: View, event: str, payload: bytes) -> View:
    """Fold one event into the panel's state.

    A metric moves a gauge, a phase moves a node's checklist, a console line goes
    to its node's tail (and its task's output, when it says whose it is), and the
    state events move the node and the pool. Everything else the panel does not
    draw is left to scroll past in the log.
    """
    if event == "node.metrics":
        reading = msgspec.json.decode(payload, type=_Reading)
        return _sample(view, reading.node, reading.name, reading.value)

    if event == "compute.cost":
        accrued = msgspec.json.decode(payload, type=_Accrued)
        return replace(view, pool=replace(view.pool, cost=accrued.cost))

    data = msgspec.json.decode(payload, type=dict[str, str])
    match event.split(".", 1):
        case ["node", "phase"]:
            return _phased(view, data)
        case ["node", "console"]:
            return _spoken(view, data)
        case ["node", state]:
            return _transition(view, data.get("node", ""), state, data.get("error"))
        case ["compute", state] if state in _STATE_STYLE:
            return replace(view, pool=replace(view.pool, state=state))
        case _:
            return view


def refresh(view: View, compute: Compute, nodes: Page[Node]) -> View:
    """Replace the structural half from the views, keeping what only the stream gave.

    The poll knows the address and the price; it does not know the utilisation, the
    phases or the tail, which never went through the store's views. Those are
    carried across by id — the poll is the base, the stream is the overlay.
    """
    spec = compute.spec.specs[0] if compute.spec.specs else None
    pool = Pool(
        name=compute.name or compute.id,
        provider=(spec.provider.kind.upper() if spec else ""),
        region=spec.region if spec else None,
        accelerator=spec.accelerator if spec else None,
        allocation=compute.spec.allocation,
        state=compute.status.state,
        ready=compute.status.nodes_ready,
        total=compute.status.nodes_total,
        cost=view.pool.cost,
    )
    kept = {node.id: node for node in view.nodes}

    def stamps(node: Node) -> Mapping[str, datetime]:
        """The stream's clock, plus a stamp for a state the poll saw first."""
        known = kept[node.id].state_at if node.id in kept else MappingProxyType({})
        if node.state in known:
            return known
        return MappingProxyType({**known, node.state: datetime.now(UTC)})

    rows = tuple(
        NodeRow(
            id=node.id,
            state=node.state,
            rank=node.rank,
            machine=node.machine,
            address=node.address,
            accelerator=node.accelerator,
            price=node.price_per_hour,
            region=node.provider_binding.get("region"),
            error=node.last_error.message if node.last_error else None,
            created_at=node.created_at,
            state_at=stamps(node),
            metrics=kept[node.id].metrics if node.id in kept else MappingProxyType({}),
            phases=kept[node.id].phases if node.id in kept else (),
            tail=kept[node.id].tail if node.id in kept else (),
            installed=kept[node.id].installed if node.id in kept else MappingProxyType({}),
            fetching=kept[node.id].fetching if node.id in kept else MappingProxyType({}),
        )
        for node in nodes.items
        if node.state != "deleted"
    )
    image = plugins.image(compute.spec.image, plugins.resolve(compute.spec.plugins))
    return replace(view, pool=pool, nodes=rows, plan=tuple(image.pip), steps=_steps(image))


def _steps(image: Image) -> tuple[str, ...]:
    """The phases the bootstrap script will run for this image, in the script's order.

    Derived from the same transformed image the connector ships, so the inspector can
    show the road ahead — ``○ deps`` — before the first phase event ever arrives.
    """
    return tuple(
        name
        for name, wanted in (
            ("shell_vars", bool(image.shell_vars)),
            ("env", bool(image.env)),
            ("apt", True),
            ("uv", True),
            ("venv", True),
            ("skyward", True),
            ("deps", bool(image.pip)),
        )
        if wanted
    )


def refresh_tasks(view: View, tasks: Page[Task], names: Mapping[str, str]) -> View:
    """Replace the task half from the poll, naming each by its function where known.

    A task carries the function's hash, not its name; the name is a second read,
    cached, so a section nobody is looking at costs nothing. The args are cloudpickle
    and stay opaque — the row says what ran and where, not with which arguments.
    """
    rows = tuple(_task_row(task, names) for task in tasks.items)
    return replace(view, tasks=rows)


def command(view: View, key: Key, width: int = 120, height: int = 36) -> View:
    """Apply one keypress. Pure: the terminal is the caller's problem, not this one's.

    The width and height are the terminal's, because up and down move by one row of
    the wall — how many blocks a row holds is a fact about the grid being drawn.
    A find in progress swallows the keyboard: digits build the rank, ``enter``
    jumps, ``esc`` cancels.
    """
    ui = view.ui
    rows = _active(view)
    if ui.find is not None:
        match key:
            case "enter":
                jumped = _jump(view, ui.find)
                return replace(jumped, ui=replace(jumped.ui, find=None))
            case "escape":
                return replace(view, ui=replace(ui, find=None))
            case "backspace":
                return replace(view, ui=replace(ui, find=ui.find[:-1]))
            case ("char", char) if char.isdigit():
                return replace(view, ui=replace(ui, find=ui.find + char))
            case _:
                return view

    cols = plan_wall(max(1, len(rows)), *_wall_area(width, height)).cols
    cursor = min(ui.cursor, max(0, len(rows) - 1))
    match key:
        case ("char", "q"):
            return replace(view, ui=replace(ui, quitting=True))
        case ("char", "/"):
            return replace(view, ui=replace(ui, find=""))
        case ("char", "a") if rows:
            return replace(view, ui=replace(ui, cursor=_next_attention(rows, cursor)))
        case "left" if rows:
            return replace(view, ui=replace(ui, cursor=max(0, cursor - 1)))
        case "right" if rows:
            return replace(view, ui=replace(ui, cursor=min(len(rows) - 1, cursor + 1)))
        case "up" if rows:
            return replace(view, ui=replace(ui, cursor=cursor - cols if cursor >= cols else cursor))
        case "down" if rows:
            return replace(view, ui=replace(ui, cursor=cursor + cols if cursor + cols < len(rows) else cursor))
        case _:
            return view


def _next_attention(rows: list[NodeRow], cursor: int) -> int:
    """The next block after the cursor that needs a human, wrapping around."""
    order = [*range(cursor + 1, len(rows)), *range(0, cursor + 1)]
    return next((i for i in order if rows[i].state in _BROKEN or _idle(rows[i])), cursor)


def _idle(row: NodeRow) -> bool:
    """Ready, with a gauge that says nothing is running on it."""
    gpu = _latest(row.metrics, "gpu_util")
    return row.state == "ready" and gpu is not None and gpu < 5.0


def _jump(view: View, query: str) -> View:
    rows = _active(view)
    match = next((i for i, row in enumerate(rows) if query and str(row.rank) == query), None)
    if match is None:
        return view
    return replace(view, ui=replace(view.ui, cursor=match))


def _sample(view: View, node_id: str, name: str, value: float) -> View:
    if name not in _METRICS or not node_id:
        return view

    def add(row: NodeRow) -> NodeRow:
        history = (*row.metrics.get(name, ()), value)[-HISTORY:]
        return replace(row, metrics=MappingProxyType({**row.metrics, name: history}))

    return _amend(view, node_id, add)


def _phased(view: View, data: dict[str, str]) -> View:
    """One phase turning over. The script's own `bootstrap` wraps all the others — the
    checklist shows the parts, so the wrapper is dropped here and never stored."""
    node_id, name = data.get("node", ""), data.get("phase", "")
    if not node_id or not name or name == "bootstrap":
        return view
    at = datetime.fromisoformat(data["at"]) if "at" in data else datetime.now(UTC)

    def mark(row: NodeRow) -> NodeRow:
        marks = {phase.name: phase for phase in row.phases}
        match data.get("event"):
            case "started":
                marks[name] = PhaseMark(name, started=at)
            case "completed":
                marks[name] = replace(marks.get(name, PhaseMark(name)), finished=at)
            case "failed":
                marks[name] = replace(marks.get(name, PhaseMark(name)), finished=at, error=data.get("error") or "failed")
        return replace(row, phases=tuple(marks.values()))

    return _amend(view, node_id, mark)


def _spoken(view: View, data: dict[str, str]) -> View:
    node_id, content = data.get("node", ""), data.get("content", "")
    if not node_id:
        return view
    tailed = _amend(view, node_id, lambda row: _heard(row, content))
    if task := data.get("task"):
        lines = (*tailed.outputs.get(task, ()), content)[-OUTPUT:]
        return replace(tailed, outputs=MappingProxyType({**tailed.outputs, task: lines}))
    return tailed


def _heard(row: NodeRow, content: str) -> NodeRow:
    """A console line lands in the tail, and the two shapes uv speaks move the packages.

    ``+ name==version`` is uv confirming an install; ``Downloading name (size)`` and
    ``Downloaded name`` are it fetching one. That is the whole package tracker — no
    new event on the wire, just the console read for what it already says.
    """
    row = replace(row, tail=(*row.tail, content)[-TAIL:])
    match content.split():
        case ["+", spec] if "==" in spec:
            name, _, version = spec.partition("==")
            return replace(row, installed=MappingProxyType({**row.installed, _bare(name): version}))
        case ["Downloading" | "Downloaded", name, *rest]:
            size = rest[0].strip("()") if rest else row.fetching.get(_bare(name), "")
            return replace(row, fetching=MappingProxyType({**row.fetching, _bare(name): size}))
        case _:
            return row


def _bare(spec: str) -> str:
    """The name a requirement scopes to: version, extras and markers dropped, normalised."""
    return re.split(r"[<>=!~ \[@;]", spec.strip(), maxsplit=1)[0].lower().replace("_", "-")


def _transition(view: View, node_id: str, state: str, error: str | None) -> View:
    if state not in _STATE_STYLE or not node_id:
        return view

    def turn(row: NodeRow) -> NodeRow:
        stamped = row.state_at if state in row.state_at else MappingProxyType({**row.state_at, state: datetime.now(UTC)})
        return replace(row, state=state, error=error or row.error, state_at=stamped)

    return _amend(view, node_id, turn)


def _amend(view: View, node_id: str, change: Callable[[NodeRow], NodeRow]) -> View:
    rows = view.nodes
    if not any(row.id == node_id for row in rows):
        rows = (*rows, NodeRow(id=node_id))
    return replace(view, nodes=tuple(change(row) if row.id == node_id else row for row in rows))


def _task_row(task: Task, names: Mapping[str, str]) -> TaskRow:
    last = task.executions[-1] if task.executions else None
    return TaskRow(
        id=task.id,
        state=task.state,
        function=names.get(task.function) or task.function[:8],
        node=last.node_id if last else None,
        submitted_at=task.submitted_at,
        started_at=last.started_at if last else None,
        finished_at=task.finished_at,
        retries=max(0, len(task.executions) - 1),
        error=last.error.message if last and last.error else None,
    )


def _active(view: View) -> list[NodeRow]:
    """The wall, in wall order: by rank, so a block never changes house."""
    return sorted((row for row in view.nodes if row.state != "deleted"), key=lambda row: row.rank)


def _focused(view: View) -> NodeRow | None:
    rows = _active(view)
    cursor = min(view.ui.cursor, len(rows) - 1)
    return rows[cursor] if rows else None


def _label(row: NodeRow) -> str:
    return f"node #{row.rank}"


def _alias(view: View, node_id: str | None) -> str | None:
    """The rank-shaped name a node id goes by on screen, when the view knows it."""
    row = next((row for row in view.nodes if row.id == node_id), None)
    return _label(row) if row else node_id


# --------------------------------------------------------------------------- #
# the wall: a honeycomb, one hexagon per node, sized by how many must fit       #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Hexes:
    """A honeycomb plan: pointy-top hexagons, odd bands half-shifted, walls shared."""

    cols: int
    scale: int


@dataclass(frozen=True, slots=True)
class Flats:
    """The fallback for pools too big for hexagons: one-line blocks, as before."""

    cols: int
    block_w: int
    numbered: bool


type Wall = Hexes | Flats


_HEX_LADDER = ((3, 6), (2, 24), (1, 96))
"""Hexagon scales, largest first, each with the most nodes it suits.

The ceiling is what makes the wall responsive to the pool rather than to the
terminal: ten nodes on a huge screen get medium hexes, not billboards. The
terminal then only shrinks the choice — a rung that does not fit steps down.
"""

_FLAT_LADDER = ((5, 192), (3, 0), (2, 0))
"""Below the smallest hexagon: flat block widths, with the same ceilings idea."""


def _hex_span(scale: int) -> tuple[int, int]:
    """A hexagon's cell box: width, and the rows from its apex to the next band's."""
    return 6 * scale, scale + (3 * scale) // 2


def plan_wall(n: int, width: int, height: int) -> Wall:
    """The cell size ``n`` nodes deserve, stepped down until everyone fits.

    The last rung always fits by decree — a terminal too small for the pixels is a
    terminal too small for the panel, and the fallback below this is the line console.
    """
    for scale, suits in _HEX_LADDER:
        if suits and n > suits:
            continue
        w, stride_y = _hex_span(scale)
        cols = min(n, max(1, (width - 1 - w // 2) // w))
        bands = -(-n // cols)
        if (bands - 1) * stride_y + 2 * scale + (3 * scale) // 2 <= height:
            return Hexes(cols, scale)
    for block_w, suits in _FLAT_LADDER:
        if suits and n > suits:
            continue
        cols = min(n, max(1, (width + 1) // (block_w + 1)))
        if -(-n // cols) <= height or block_w == 2:
            return Flats(cols, block_w, numbered=block_w >= len(str(n - 1)) + 2)
    raise AssertionError("the ladder's last rung always fits")


def _wall_area(width: int, height: int) -> tuple[int, int]:
    """What is left for the wall once the inspector and the chrome take their share."""
    return max(8, width - INSPECTOR - 4), max(3, height - 8)


_TONE_BG_DARK = (16, 20, 26)
_TONE_BG_LIGHT = (250, 250, 248)

_STAGE_RGB = {
    "provision": (201, 162, 63),
    "connect": (67, 169, 191),
    "bootstrap": (91, 131, 224),
    "ready": (63, 174, 118),
    "drain": (201, 162, 63),
    "gone": (110, 118, 129),
    "lost": (214, 84, 79),
    "failed": (214, 84, 79),
    "degraded": (214, 84, 79),
}
"""A stage's colour on the wall, before it is toned against the terminal's background."""


def _tone(rgb: tuple[int, int, int], strength: float, bg: tuple[int, int, int]) -> str:
    """The stage colour mixed toward the terminal's own background — opacity, in cells."""
    r, g, b = (round(c * strength + back * (1 - strength)) for c, back in zip(rgb, bg, strict=True))
    return f"#{r:02x}{g:02x}{b:02x}"


def _strength(row: NodeRow) -> float:
    """How lit a block is: a ready node glows with its GPU, everything else is loud."""
    if row.state != "ready":
        return 1.0 if row.state in _BROKEN else 0.9
    match _latest(row.metrics, "gpu_util"):
        case None:
            return 0.8
        case float(gpu) if gpu < 5:
            return 0.4
        case float(gpu) if gpu < 40:
            return 0.6
        case float(gpu) if gpu < 75:
            return 0.8
        case _:
            return 1.0


def _block_colors(row: NodeRow, bg: tuple[int, int, int]) -> tuple[str, str]:
    """The block's ink and fill, the fill already toned against the terminal."""
    fill = _tone(_STAGE_RGB.get(_STAGE.get(row.state, "gone"), (110, 118, 129)), _strength(row), bg)
    r, g, b = int(fill[1:3], 16), int(fill[3:5], 16), int(fill[5:7], 16)
    ink = "#ffffff" if 0.299 * r + 0.587 * g + 0.114 * b < 150 else "#20242c"
    return ink, fill


_TOP_LEFT = "🭈🭆🭂"
_TOP_RIGHT = "🭍🭑🬽"
_BOTTOM_LEFT = "🭣🭧🭓"
_BOTTOM_RIGHT = "🭞🭜🭘"
"""Legacy-computing wedges: quarter-cell diagonals the terminal draws procedurally.

Chained over three columns they make the ~30° shoulder of a pointy-top hexagon;
where two hexagons meet, the shared cell is one wedge with the upper neighbour's
fill behind it — the honeycomb's walls are shared, not adjacent.
"""

type _Cell = tuple[str, str | None, str | None, bool]
"""One canvas cell: glyph, foreground fill, background fill, underline."""


def _wall(view: View, plan: Wall, bg: tuple[int, int, int]) -> RenderableType:
    """Every node as one cell of the honeycomb — rank inside, the cursor a lifted shade."""
    rows = _active(view)
    if not rows:
        return Text("  waiting for the first machine…", style="grey50")
    cursor = min(view.ui.cursor, len(rows) - 1)
    match plan:
        case Hexes():
            return _hex_wall(rows, cursor, plan, bg)
        case Flats():
            lines: list[Text] = []
            for start in range(0, len(rows), plan.cols):
                text = Text(" ")
                for offset, row in enumerate(rows[start : start + plan.cols]):
                    text.append(" ")
                    text.append_text(_flat_block(row, plan, selected=start + offset == cursor, bg=bg))
                lines.append(text)
            return Group(*lines)


def _hex_wall(rows: list[NodeRow], cursor: int, plan: Hexes, bg: tuple[int, int, int]) -> RenderableType:
    canvas: dict[tuple[int, int], _Cell] = {}
    w, stride_y = _hex_span(plan.scale)
    for i, row in enumerate(rows):
        band, col = divmod(i, plan.cols)
        x0 = 1 + col * w + (band % 2) * (w // 2)
        _paint_hex(canvas, x0, band * stride_y, plan.scale, row, selected=i == cursor, bg=bg)
    lines: list[Text] = []
    for r in range(max(r for r, _ in canvas) + 1):
        text = Text()
        for c in range(max(c for _, c in canvas) + 1):
            ch, fg, fill, underline = canvas.get((r, c), (" ", None, None, False))
            text.append(ch, style=Style(color=fg, bgcolor=fill, underline=underline, bold=ch.isdigit()))
        lines.append(text)
    return Group(*lines)


def _paint_hex(
    canvas: dict[tuple[int, int], _Cell],
    x0: int,
    y0: int,
    scale: int,
    row: NodeRow,
    selected: bool,
    bg: tuple[int, int, int],
) -> None:
    """One pointy-top hexagon: wedge rows, solid rows, wedge rows — the rank mid-way.

    Focus is quiet on purpose: the fill brightens rather than gaining a frame, and
    the rank underlines. The inspector's title says the same thing in words.
    """
    ink, fill = _block_colors(row, bg)
    if selected:
        fill = _lift(fill)
    w = 6 * scale
    rows_tall = 2 * scale + (3 * scale) // 2

    def wedges(r: int, c0: int, chars: str) -> None:
        for i, ch in enumerate(chars):
            key = (y0 + r, x0 + c0 + i)
            old = canvas.get(key)
            canvas[key] = (ch, fill, old[1] if old else None, False)

    def solid(r: int, c0: int, c1: int) -> None:
        for c in range(c0, c1 + 1):
            canvas[(y0 + r, x0 + c)] = (" ", None, fill, False)

    for r in range(scale):
        indent = 3 * (scale - 1 - r)
        wedges(r, indent, _TOP_LEFT)
        solid(r, indent + 3, w - indent - 4)
        wedges(r, w - indent - 3, _TOP_RIGHT)
        bottom = rows_tall - 1 - r
        wedges(bottom, indent, _BOTTOM_LEFT)
        solid(bottom, indent + 3, w - indent - 4)
        wedges(bottom, w - indent - 3, _BOTTOM_RIGHT)
    for r in range(scale, rows_tall - scale):
        solid(r, 0, w - 1)
    label = str(row.rank)
    start = (w - len(label)) // 2
    for i, ch in enumerate(label):
        canvas[(y0 + (rows_tall - 1) // 2, x0 + start + i)] = (ch, ink, fill, selected)


def _flat_block(row: NodeRow, plan: Flats, selected: bool, bg: tuple[int, int, int]) -> Text:
    ink, fill = _block_colors(row, bg)
    if selected:
        fill = _lift(fill)
    style = f"bold {ink} on {fill}"
    label = str(row.rank) if plan.numbered else ""
    text = Text(label.center(plan.block_w), style=style)
    if selected and label:
        start = (plan.block_w - len(label)) // 2
        text.stylize("underline", start, start + len(label))
    return text


def _lift(fill: str, amount: float = 0.24) -> str:
    """The fill, a shade closer to white — how the cursor rests on a block."""
    r, g, b = (int(fill[i : i + 2], 16) for i in (1, 3, 5))
    lifted = (round(c + (255 - c) * amount) for c in (r, g, b))
    return "#" + "".join(f"{c:02x}" for c in lifted)


# --------------------------------------------------------------------------- #
# the inspector: the node under the cursor, same sections at every zoom         #
# --------------------------------------------------------------------------- #


def _inspector(view: View, accent: str) -> RenderableType:
    row = _focused(view)
    if row is None:
        body: RenderableType = Text("nothing to inspect yet", style="grey50")
        title = "inspector"
    else:
        body = Group(*_sections(view, row))
        title = f"inspector · #{row.rank}"
    return Panel(
        body,
        title=Text(title.upper(), style="grey50"),
        title_align="left",
        border_style=_FAINT,
        width=INSPECTOR,
        padding=(0, 1),
    )


def _sections(view: View, row: NodeRow) -> list[RenderableType]:
    parts: list[RenderableType] = [_head(row)]
    if row.state in _BROKEN or row.state in _PRE_READY:
        parts += [_section("journey"), *_journey(row)]
        if error := _first_error(row):
            parts.append(_callout(error))
        if row.state in _PRE_READY and row.error:
            parts += [_section("retry"), _kv(Text.assemble((f"{_spin()} ", "yellow"), (_stage_word(row.state), "yellow")), _stage_clock(row))]
    if row.state == "ready":
        parts += [_section("metrics"), *_meter_rows(row)]
        if work := _work_rows(view, row):
            parts += [_section("work"), *work]
    if machine := _machine_rows(view, row):
        parts += [_section("machine"), *machine]
    if row.state != "ready" and row.tail:
        parts += [_section("output"), *(Text(line, style="grey50", overflow="ellipsis", no_wrap=True) for line in row.tail[-3:])]
    return parts


def _head(row: NodeRow) -> RenderableType:
    word = _stage_word(row.state)
    rgb = _STAGE_RGB.get(_STAGE.get(row.state, "gone"), (110, 118, 129))
    pill = Text()
    pill.append(f" {_glyph(row)} {word} ", style=f"bold #ffffff on #{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}")
    return _kv(pill, _dim(_uptime(row.created_at)))


def _glyph(row: NodeRow) -> str:
    if row.state in _BROKEN:
        return "✗"
    return _spin() if row.state in _PRE_READY else "●"


def _stage_word(state: str) -> str:
    return _STAGE.get(state, state)


def _stage_clock(row: NodeRow) -> Text:
    entered = row.state_at.get(row.state) or row.created_at
    return Text(_clock(_since(entered, None)), style="bold")


def _journey(row: NodeRow) -> list[RenderableType]:
    """The four stages with what each one cost, the current one still counting."""
    marks = _journey_marks(row)
    lines: list[RenderableType] = []
    for name, glyph, style, clock in marks:
        left = Text.assemble((f"{glyph} ", style), (name, "default" if style != _FAINT else _FAINT))
        lines.append(_kv(left, Text(clock, style="bold" if glyph == _spin() else "grey50")))
    return lines


def _journey_marks(row: NodeRow) -> list[tuple[str, str, str, str]]:
    at = row.state_at
    born = at.get("provisioning") or at.get("requested") or row.created_at
    stages = (
        ("provision", born, at.get("connecting")),
        ("connect", at.get("connecting"), at.get("bootstrapping")),
        ("bootstrap", at.get("bootstrapping"), at.get("ready")),
        ("ready", at.get("ready"), None),
    )
    current = _STAGE.get(row.state, "")
    failed_at = None
    if row.state in _BROKEN:
        current = ""
        reached = [name for name, started, _ in stages if started is not None]
        failed_at = reached[-1] if reached else "provision"

    marks: list[tuple[str, str, str, str]] = []
    for name, started, ended in stages:
        clock = _clock(_since(started, ended)) if started else "—"
        if name == failed_at:
            marks.append((name, "✗", "red", clock))
        elif name == current:
            doing = f"{name} · {phase}" if name == "bootstrap" and (phase := _running_phase(row)) else name
            marks.append((doing, _spin(), _STATE_STYLE.get(row.state, "blue"), _clock(_since(started, None)) or "0:00"))
        elif ended is not None or (started is not None and name != "ready"):
            marks.append((name, "✓", "green", clock))
        elif name == "ready" and started is not None:
            marks.append((name, "●", "green", ""))
        else:
            marks.append((name, "○", _FAINT, "—"))
    return marks


def _first_error(row: NodeRow) -> str | None:
    failed = next((mark.error for mark in row.phases if mark.error), None)
    return failed or (row.error if row.state in _BROKEN or row.state in _PRE_READY and row.error else None)


def _callout(error: str) -> Text:
    return Text.assemble(("▎", "red"), (error, "red")).copy()


def _meter_rows(row: NodeRow) -> list[RenderableType]:
    gpu, vram, cpu, mem = _gauges(row.metrics)
    rows = [
        (name, value, spark)
        for name, value, spark in (
            ("gpu", _pct_text(gpu), _spark(row.metrics.get("gpu_util", ()), 100.0)),
            ("vram", _pair_text(vram), _spark(row.metrics.get("gpu_mem_mb", ()), vram[1] if vram else None)),
            ("cpu", _pct_text(cpu), _spark(row.metrics.get("cpu", ()), 100.0)),
            ("mem", _pair_text(mem), _spark(row.metrics.get("mem_used_mb", ()), mem[1] if mem else None)),
        )
        if value is not None
    ]
    if not rows:
        return [Text("no readings yet", style="grey50")]
    return [_kv(_dim(name), Text.assemble(value, "  ", spark)) for name, value, spark in rows]


def _work_rows(view: View, row: NodeRow) -> list[RenderableType]:
    """What this node is doing for a living, from the task half of the poll."""
    mine = [task for task in view.tasks if task.node == row.id]
    if not mine:
        return []
    lines: list[RenderableType] = []
    if running := next((task for task in mine if task.state == "running"), None):
        clock = _clock(_since(running.started_at or running.submitted_at, None))
        lines.append(_kv(_dim("running"), Text.assemble((running.function, "magenta"), (f" {clock}", "grey50"))))
    done = [task for task in mine if task.state == "succeeded"]
    if done:
        took = [seconds for task in done if (seconds := _since(task.started_at, task.finished_at)) is not None]
        avg = f" · avg {_clock(round(sum(took) / len(took)))}" if took else ""
        lines.append(_kv(_dim("done here"), _dim(f"{len(done)}{avg}")))
    if failed := next((task for task in mine if task.error), None):
        lines.append(_callout(failed.error or "task failed"))
    return lines


def _machine_rows(view: View, row: NodeRow) -> list[RenderableType]:
    pairs = [
        ("instance", row.machine),
        ("type", row.accelerator),
        ("address", row.address),
        ("zone", row.region),
        ("price", f"${row.price:,.2f}/hr" if row.price else None),
    ]
    lines: list[RenderableType] = [_kv(_dim(key), Text(value)) for key, value in pairs if value]
    if _idle(row):
        lines.append(_kv(_dim("idle"), Text(_idle_cost(view, row), style="yellow")))
    return lines


def _idle_cost(view: View, row: NodeRow) -> str:
    ended = [task.finished_at for task in view.tasks if task.node == row.id and task.finished_at]
    since = max(ended) if ended else row.state_at.get("ready")
    seconds = _since(since, None)
    if seconds is None or row.price is None:
        return "now"
    return f"{_clock(seconds)} · ≈ ${row.price * seconds / 3600:,.2f}"


def _section(title: str) -> Text:
    line = Text.assemble((title.upper(), "grey50"), (" " + "─" * max(0, INSPECTOR - 6 - len(title)), _LINE))
    line.truncate(INSPECTOR - 4)
    return line


def _kv(left: Text, right: Text) -> Table:
    grid = Table.grid(expand=True)
    grid.add_column(no_wrap=True, overflow="ellipsis")
    grid.add_column(justify="right", no_wrap=True, overflow="ellipsis")
    grid.add_row(left, right)
    return grid


# --------------------------------------------------------------------------- #
# render                                                                        #
# --------------------------------------------------------------------------- #


def _terminal_background() -> tuple[int, int, int] | None:
    """Ask the terminal its background colour (OSC 11); None when it will not say."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None
    fd = sys.stdin.fileno()
    try:
        saved = termios.tcgetattr(fd)
    except (termios.error, ValueError, OSError):
        return None
    try:
        tty.setraw(fd)
        os.write(sys.stdout.fileno(), b"\033]11;?\033\\")
        if not select.select([fd], [], [], 0.2)[0]:
            return None
        reply = b""
        while select.select([fd], [], [], 0.05)[0]:
            reply += os.read(fd, 1024)
        decoded = reply.decode("latin-1")
        if "rgb:" not in decoded:
            return None
        channels = decoded.split("rgb:")[1].split("\033")[0].split("\\")[0].split("/")
        if len(channels) != 3:
            return None
        r, g, b = (int(channel[:2], 16) for channel in channels)
        return r, g, b
    except (OSError, ValueError):
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


_BACKGROUND = _terminal_background()


def _dark_terminal() -> bool:
    """Dark unless the terminal says otherwise — the safe guess for a terminal."""
    match _BACKGROUND:
        case (r, g, b):
            return 0.299 * r + 0.587 * g + 0.114 * b < 128
        case _:
            return True


_DARK = _dark_terminal()

_FAINT = "grey35" if _DARK else "grey66"
_LINE = "grey23" if _DARK else "grey78"
_ACCENT_HEX = "#4d8cff" if _DARK else "#1d6fd8"
_CAP = "grey74 on grey23" if _DARK else "grey19 on grey85"
"""A keycap in the footer."""

_WALL_BG = _BACKGROUND or (_TONE_BG_DARK if _DARK else _TONE_BG_LIGHT)


def render(view: View, width: int = 120, height: int = 36) -> RenderableType:
    """The pinned panel: a rule against the log, the strip, the wall beside the inspector, the keys."""
    plan = plan_wall(max(1, len(_active(view))), *_wall_area(width, height))
    body = Table.grid(padding=(0, 1), expand=True)
    body.add_column(ratio=1)
    body.add_column(width=INSPECTOR)
    body.add_row(
        Group(_wall(view, plan, _WALL_BG), Text(""), _counts(view)),
        _inspector(view, _ACCENT_HEX),
    )
    return Group(Rule(style=_LINE), _strip(view), Text(""), body, _keys(view))


def _strip(view: View) -> Table:
    """The pool in one line: the spec as the name, the money and the count at the edge."""
    pool = view.pool
    spec = " · ".join(
        part
        for part in (
            f"{pool.total}× {pool.accelerator}" if pool.accelerator and pool.total else pool.accelerator,
            pool.provider,
            pool.region,
            pool.allocation,
        )
        if part
    )
    left = Text.assemble(
        ("● ", _STATE_STYLE.get(pool.state, "grey50")),
        (spec or pool.name or "pool", "bold"),
        *(("   ", _dim(pool.name)) if spec and pool.name else ()),
    )
    running = sum(1 for task in view.tasks if task.state == "running")
    prices = [row.price for row in _active(view) if row.price]
    right = " · ".join(
        part
        for part in (
            f"{pool.ready}/{pool.total} ready" if pool.total else "",
            f"{running} running" if running else "",
            f"${sum(prices):,.2f}/hr" if prices else "",
            f"${pool.cost:,.2f}" if pool.cost is not None else "",
        )
        if part
    )
    grid = Table.grid(expand=True)
    grid.add_column()
    grid.add_column(justify="right")
    grid.add_row(left, _dim(right))
    return grid


def _counts(view: View) -> Text:
    """The wall, totalled: every stage present with its count, idle called out in money terms."""
    rows = _active(view)
    if not rows:
        return Text("")
    styles = {
        "provision": "yellow",
        "connect": "cyan",
        "bootstrap": "blue",
        "ready": "green",
        "drain": "yellow",
        "gone": "grey50",
        "lost": "red",
        "failed": "red",
        "degraded": "red",
    }
    stages = Counter(_stage_word(row.state) for row in rows)
    text = Text("  ")
    for i, (stage, count) in enumerate(sorted(stages.items(), key=lambda item: -item[1])):
        if i:
            text.append(" · ", style=_FAINT)
        text.append(f"{stage} ", style="grey50")
        text.append(str(count), style=f"bold {styles.get(stage, 'default')}")
    if idle := sum(1 for row in rows if _idle(row)):
        text.append(" · ", style=_FAINT)
        text.append(f"idle {idle}", style="bold yellow")
    return text


def _keys(view: View) -> Text:
    """The footer as keycaps: what the keyboard can do, and nothing it cannot."""
    ui = view.ui
    if ui.find is not None:
        return Text.assemble(
            ("go to ", "grey50"), ("#", _ACCENT_HEX), (ui.find, "default"), ("▏", _ACCENT_HEX), _dim("   enter jump · esc cancel"),
        )

    def cap(label: str, hint: str) -> tuple[Text, ...]:
        return (Text(f" {label} ", style=_CAP), _dim(f" {hint}  "))

    rows = _active(view)
    parts = [*cap("←↑↓→", "move")]
    if any(row.state in _BROKEN or _idle(row) for row in rows):
        parts += cap("a", "next attention")
    if len(rows) > 9:
        parts += cap("/", "go to rank")
    parts += cap("q", "quit")
    return Text.assemble(*parts)


# --------------------------------------------------------------------------- #
# render helpers                                                                #
# --------------------------------------------------------------------------- #


def _gauges(
    metrics: Mapping[str, tuple[float, ...]],
) -> tuple[float | None, tuple[float, float] | None, float | None, tuple[float, float] | None]:
    gpu = _latest(metrics, "gpu_util")
    cpu = _latest(metrics, "cpu")
    vram = _pair(_latest(metrics, "gpu_mem_mb"), _latest(metrics, "gpu_mem_total_mb"))
    mem = _pair(_latest(metrics, "mem_used_mb"), _latest(metrics, "mem_total_mb"))
    return gpu, vram, cpu, mem


def _latest(metrics: Mapping[str, tuple[float, ...]], name: str) -> float | None:
    history = metrics.get(name)
    return history[-1] if history else None


def _pair(used: float | None, total: float | None) -> tuple[float, float] | None:
    if used is None or total is None or total <= 0:
        return None
    return used, total


def _pct_text(pct: float | None) -> Text | None:
    return Text(f"{pct:.0f}%", style="bold") if pct is not None else None


def _pair_text(pair: tuple[float, float] | None) -> Text | None:
    if pair is None:
        return None
    used, total = pair
    return Text.assemble((_gb(used), "bold"), (f"/{_gb(total)}G", "grey50"))


def _gb(mb: float) -> str:
    """Megabytes as gigabytes, with a decimal only while it is small enough to need one."""
    gb = mb / 1024
    return f"{gb:.0f}" if gb >= 10 else f"{gb:.1f}"


def _spark(history: tuple[float, ...], top: float | None) -> Text:
    """The last readings as blocks, scaled to the gauge's own ceiling."""
    if not history or not top:
        return Text("")
    blocks = "▁▂▃▄▅▆▇█"

    def block(value: float) -> str:
        frac = max(0.0, min(1.0, value / top))
        return blocks[round(frac * (len(blocks) - 1))]

    return Text("".join(block(value) for value in history), style="cyan" if _DARK else "dark_cyan")


def _spin() -> str:
    """The frame the clock says it is. A function of time, so every spinner turns together."""
    return SPINNER[int(time.monotonic() * 10) % len(SPINNER)]


def _running_phase(row: NodeRow) -> str | None:
    return next((mark.name for mark in reversed(row.phases) if mark.started and not mark.finished), None)


def _since(start: datetime | None, end: datetime | None) -> int | None:
    if start is None:
        return None
    started = start if start.tzinfo else start.replace(tzinfo=UTC)
    finished = end if end and end.tzinfo else (end.replace(tzinfo=UTC) if end else datetime.now(UTC))
    seconds = int((finished - started).total_seconds())
    return seconds if seconds >= 0 else None


def _clock(seconds: int | None) -> str:
    if seconds is None:
        return ""
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _uptime(created: datetime | None) -> str:
    clock = _clock(_since(created, None))
    return f"up {clock}" if clock else ""


def _dim(text: str) -> Text:
    return Text(text, style="grey50")


# --------------------------------------------------------------------------- #
# keyboard: raw bytes → keys, and the terminal put in cbreak while the panel is up #
# --------------------------------------------------------------------------- #


def _decode(text: str) -> list[Key]:
    """The bytes a cbreak read handed back, split into keys.

    Arrows arrive as a three-byte escape sequence and a lone ``esc`` as one byte;
    the difference is whether a ``[`` follows. Everything printable is a character,
    which is what a rank query is made of.
    """
    keys: list[Key] = []
    i = 0
    while i < len(text):
        char = text[i]
        if char == "\x1b" and text[i + 1 : i + 2] == "[" and i + 2 < len(text):
            match text[i + 2]:
                case "A":
                    keys.append("up")
                case "B":
                    keys.append("down")
                case "C":
                    keys.append("right")
                case "D":
                    keys.append("left")
                case _:
                    keys.append("escape")
            i += 3
            continue
        if char == "\x1b":
            keys.append("escape")
        elif char == "\t":
            keys.append("tab")
        elif char in "\r\n":
            keys.append("enter")
        elif char in "\x7f\x08":
            keys.append("backspace")
        elif char.isprintable():
            keys.append(("char", char))
        i += 1
    return keys


@contextmanager
def _keyboard(fd: int, loop: asyncio.AbstractEventLoop, on_key: Callable[[], None]) -> Iterator[bool]:
    """Put the terminal in cbreak and wake ``on_key`` when a key is pressed.

    Cbreak, not raw: the ``ISIG`` bit stays on, so ``Ctrl-C`` is still a ``SIGINT``
    delivered to the main thread — the panel reads the other keys, it does not
    swallow the one that stops the program. Whatever the state was on the way in is
    what it is put back to on the way out.
    """
    if fd < 0:
        yield False
        return
    try:
        saved = termios.tcgetattr(fd)
        tty.setcbreak(fd)
    except (termios.error, ValueError, OSError):
        yield False
        return
    try:
        loop.add_reader(fd, on_key)
    except (ValueError, OSError, NotImplementedError):
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        yield False
        return
    try:
        yield True
    finally:
        with suppress(ValueError, OSError):
            loop.remove_reader(fd)
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


@dataclass(frozen=True, slots=True)
class _Snapshot:
    """One reading of the slow half, before it is folded into the view."""

    compute: Compute
    nodes: Page[Node]
    tasks: Page[Task]
    names: Mapping[str, str]


class Dashboard:
    """The live panel, for as long as the pool is open.

    Four things move it: the events, which carry the fast half — states, phases,
    gauges, and the scrolling log — a slow poll, which re-reads the structure and
    the tasks the events never carried, the keyboard, which moves the cursor
    over the wall, and a small clock, which keeps the spinners turning between
    events. All write the same ``View`` on the one loop thread, so none of them
    locks. Painting is the one thing that leaves it: a frame is built from an
    immutable view on a thread, and a burst of changes collapses into one frame.
    """

    def __init__(self, client: Client, compute: str, out: TextIO | None = None) -> None:
        self._client = client
        self._compute = compute
        self._console = RichConsole(file=out or sys.stderr)
        self._view = View()
        self._live: Live | None = None
        self._names: dict[str, str] = {}
        self._booted: set[str] = set()
        self._done = asyncio.Event()
        self._dirty = asyncio.Event()
        try:
            self._fd = sys.stdin.fileno()
        except (ValueError, OSError):
            self._fd = -1

    async def follow(self) -> None:
        with suppress(Exception):
            self._merge(await self._fetch())
        loop = asyncio.get_running_loop()
        with (
            _quiet(self._console),
            _keyboard(self._fd, loop, self._on_key),
            Live(self._render(), console=self._console, refresh_per_second=8, transient=True) as live,
        ):
            self._live = live
            poll = asyncio.create_task(self._poll())
            spin = asyncio.create_task(self._turn())
            events = asyncio.create_task(self._events())
            paint = asyncio.create_task(self._painter())
            quit_ = asyncio.create_task(self._done.wait())
            try:
                await asyncio.wait({events, quit_}, return_when=asyncio.FIRST_COMPLETED)
                if events.done() and not events.cancelled() and events.exception() is not None:
                    self._console.print(f"skyward: the event stream stopped ({events.exception()})")
            finally:
                for task in (poll, spin, events, paint, quit_):
                    task.cancel()

    def _render(self) -> RenderableType:
        return render(self._view, self._console.size.width, self._console.size.height)

    def _on_key(self) -> None:
        try:
            data = os.read(self._fd, 256)
        except OSError:
            return
        if not data:
            return
        for key in _decode(data.decode("utf-8", "ignore")):
            self._view = command(self._view, key, self._console.size.width, self._console.size.height)
        if self._view.ui.quitting:
            self._done.set()
        self._paint()

    async def _events(self) -> None:
        async with aclosing(self._client.events(self._compute)) as stream:
            async for event, payload in stream:
                self._view = observe(self._view, event, payload)
                if event == "node.ready":
                    self._booted.add(_safe(payload).get("node", ""))
                if self._scrolls(event, payload):
                    data = _safe(payload)
                    if node := data.get("node"):
                        data = {**data, "node": _alias(self._view, node) or node}
                    if line := log_line(event, data, color=True):
                        await asyncio.to_thread(self._console.print, Text.from_ansi(line))
                self._paint()

    def _scrolls(self, event: str, payload: bytes) -> bool:
        """Whether this event earns a line above the panel, here and now.

        A node's console waits for that node's ``ready`` *in the stream* — not in
        the view, which a poll may have filled in ahead of the replay — so a
        bootstrap replayed after the fact stays in the tail where it belongs.
        """
        if event not in _SCROLLED:
            return False
        if event != "node.console":
            return True
        return _safe(payload).get("node", "") in self._booted

    async def _turn(self) -> None:
        """Repaint on the clock: fast while something is booting, gently otherwise."""
        while True:
            booting = any(row.state in _PRE_READY for row in _active(self._view))
            await asyncio.sleep(0.12 if booting else 1.0)
            self._paint()

    async def _poll(self) -> None:
        while True:
            await asyncio.sleep(POLL)
            try:
                snapshot = await self._fetch()
            except asyncio.CancelledError:
                raise
            except Exception:
                continue
            self._merge(snapshot)
            self._paint()

    async def _fetch(self) -> _Snapshot:
        """Read the slow half off the daemon, touching no view: the merge is the caller's.

        Reading ``self._view`` here and again where the result lands would open a window
        the whole width of these awaits, and a key pressed inside it — a cursor move —
        would be read into the snapshot and then written back over. So this returns
        only what the daemon said, and :meth:`_merge` folds it into whatever the view
        has become in one synchronous step the keyboard cannot interleave.
        """
        compute = await self._client.call("GET", f"/v1/computes/{self._compute}", Compute)
        nodes = await self._client.call("GET", f"/v1/computes/{self._compute}/nodes", Page[Node])
        tasks: Page[Task] = Page(items=())
        with suppress(Exception):
            tasks = await self._client.call("GET", "/v1/tasks", Page[Task], compute=self._compute, limit=200)
        return _Snapshot(compute, nodes, tasks, await self._names_for(tasks))

    def _merge(self, snapshot: _Snapshot) -> None:
        merged = refresh(self._view, snapshot.compute, snapshot.nodes)
        self._view = refresh_tasks(merged, snapshot.tasks, snapshot.names)

    async def _names_for(self, tasks: Page[Task]) -> Mapping[str, str]:
        """Fill in the function names the inspector wants, once each and then cached."""
        for sha in {task.function for task in tasks.items} - self._names.keys():
            try:
                function = await self._client.call("GET", f"/v1/functions/{sha}", Function)
                self._names[sha] = function.name or sha[:8]
            except Exception:
                self._names[sha] = sha[:8]
        return self._names

    def _paint(self) -> None:
        self._dirty.set()

    async def _painter(self) -> None:
        """Render off the loop, one frame at a time, coalescing bursts.

        Building the renderable walks the whole wall — real CPU on a large pool —
        and the loop it would otherwise run on is the one carrying the event
        stream and the lease. The view is an immutable value, so a thread renders
        the one it was handed while the loop keeps replacing it; whatever changed
        mid-frame set the flag again and becomes the next frame.
        """
        while True:
            await self._dirty.wait()
            self._dirty.clear()
            if self._live is not None:
                self._live.update(await asyncio.to_thread(self._render))


def _safe(payload: bytes) -> dict[str, str]:
    """The log line only reads strings; a metric's float never reaches it, but be sure."""
    try:
        return msgspec.json.decode(payload, type=dict[str, str])
    except msgspec.ValidationError:
        return {}


class _LogSink(logging.Handler):
    """Whatever the libraries still want to warn about, printed above the panel.

    A pinned panel and a raw ``StreamHandler`` cannot share a terminal: the handler
    writes where the panel is drawing and the two smear together. So while the panel
    is up this is the only handler on the root, and it prints through the same
    console the panel does — the one thing ``Live`` knows how to draw around.
    """

    def __init__(self, console: RichConsole) -> None:
        super().__init__(logging.WARNING)
        self._console = console
        self.setFormatter(logging.Formatter("%(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._console.print(Text(self.format(record), style="yellow"))
        except Exception:
            self.handleError(record)


@contextmanager
def _quiet(console: RichConsole) -> Iterator[None]:
    """Hold the root logger's chatter down while the panel owns the screen.

    The embedded daemon brings a web framework that turns the root logger up to
    INFO on startup — every HTTP call and SSH channel, a line each. That is fine
    scrolling past a line console; under a pinned panel it is noise landing in the
    middle of the picture, and it writes to a stderr the panel cannot lift above.
    This takes the root to WARNING for the panel's lifetime and hands it back.
    """
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    root.handlers = [_LogSink(console)]
    root.setLevel(logging.WARNING)
    try:
        yield
    finally:
        root.handlers, root.level = handlers, level
