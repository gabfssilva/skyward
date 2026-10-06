"""The compute screen: one compute, read from the API, with its nodes, its tasks and its log.

The fleet screen follows the event stream, which is right for what is happening
now. This screen is opened on one compute that may have ended last week, so it
reads instead: every :data:`~skyward.core.tui.cells.POLL` seconds it asks
:func:`skyward.core.history.snapshot` for the compute, the last ten minutes of
each node's load and the idleness derived from them, and draws everything from
that one value. A read that fails leaves the last snapshot on screen and says so
in the header; the next poll tries again. One that comes back after the screen
was closed is dropped: there is nothing left to draw it on.

The two timers do different work. The poll does the I/O and redraws all of it.
The spinner timer does none, and redraws only while something on screen is
moving. Both stop while another screen, or a dialog, is on top.

Writes — kill a node, scale, delete — are asked in a dialog and sent from a
worker, so the screen keeps repainting while the daemon answers. The one that
succeeds triggers a read at once rather than waiting for the poll.

The nodes are drawn as the Claude Code panel draws them: one block per node, a
line saying what it is and how long it has been up, its load under it, and an
arrow that opens the block on its trends — the last ten minutes of cpu and gpu as
sparklines, network, memory, temperature and power — and on its address. Any
number of blocks can be open at once, and they stay open across reads. The arrow
keys walk the blocks for a keyboard and enter opens the one under the cursor;
``l`` and ``k`` act on that one. The cursor is a band behind the block, drawn
only while the list has the focus: the band says where the keys go, and the
tasks under it have a cursor of their own.

Everything a key does can be clicked: the words that name an action on the
screen are links to it (:class:`~skyward.core.tui.cells.Links`), and they run
the action the key is bound to.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from itertools import dropwhile

from msgspec import UnsetType
from rich.style import Style
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Static
from textual.worker import Worker

from skyward.api.v1 import ComputeResource, NodeBounds, NodeResource, NodeState
from skyward.core import history, writes
from skyward.core.client import FAILURES, Client
from skyward.core.history import NODE_GROUPS, NodeGroup, Snapshot
from skyward.core.tui.cells import (
    ERROR_STYLE,
    MOVING,
    NAMED_TASKS,
    NODE_MOVING,
    NODE_STYLES,
    POLL,
    REFRESH,
    Links,
    aware,
    clock,
    dots,
    link,
    money,
    rank_badge,
    spark,
    state_badge,
)
from skyward.core.tui.dialogs import Confirm, Scale
from skyward.core.tui.log import Log
from skyward.core.tui.tasks import Tasks
from skyward.core.widgets import (
    _PHASE_LABELS,
    _SPINNER_FRAMES,
    DIM,
    WARNING_STYLE,
    _accelerator_label,
    _format_duration,
)


class ComputeScreen(Screen[None]):
    """One compute, full height: header, its nodes as blocks that open on their trends, its tasks and its log."""

    DEFAULT_CSS = """
    ComputeScreen #actions { height: auto; margin: 1 2 0 2; }
    ComputeScreen #header { height: auto; margin: 1 2 0 2; }
    ComputeScreen #groups { height: auto; margin: 1 2 0 2; }
    ComputeScreen #nodes { height: auto; margin: 0 2; }
    ComputeScreen Tasks { height: auto; margin: 1 1 0 1; }
    ComputeScreen Log { height: 1fr; min-height: 12; margin: 1 1 0 1; }
    """
    BINDINGS = [
        Binding("escape", "app.pop_screen", "back"),
        Binding("up", "move(-1)", "up", show=False),
        Binding("down", "move(1)", "down", show=False),
        Binding("enter", "expand", "expand node"),
        Binding("g", "cycle", "group"),
        Binding("l", "logs", "node log"),
        Binding("k", "kill", "kill"),
        Binding("s", "scale", "scale"),
        Binding("d", "delete", "delete"),
    ]

    def __init__(self, client: Client, compute_id: str) -> None:
        super().__init__()
        self._client = client
        self._compute = compute_id
        self._tick = 0
        self._snapshot: Snapshot | None = None
        self._failure: str | None = None
        self._group: NodeGroup | None = None
        self._selected: str | None = None
        self._expanded: frozenset[str] = frozenset()
        self._reader: Worker[None] | None = None

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="page", can_focus=False):
            yield Links(id="actions")
            yield Static(id="header")
            yield Links(id="groups")
            nodes = Links(id="nodes")
            nodes.can_focus = True
            yield nodes
            yield Tasks(self._client, self._compute, lambda: self.query_one(Log).task_id)
            yield Log(self._client, self._compute)
        yield Footer()

    def on_mount(self) -> None:
        self._spinner = self.set_interval(REFRESH, self._spin)
        self._poller = self.set_interval(POLL, self._poll)
        self._paint()
        self._read_now()

    def on_screen_suspend(self) -> None:
        self._spinner.pause()
        self._poller.pause()

    def on_screen_resume(self) -> None:
        self._spinner.resume()
        self._poller.resume()
        self._poll()

    def on_descendant_focus(self) -> None:
        self._paint()

    def on_descendant_blur(self) -> None:
        self._paint()

    async def on_links_pressed(self, message: Links.Pressed) -> None:
        await self.run_action(message.action)

    def on_tasks_logs(self, message: Tasks.Logs) -> None:
        log = self.query_one(Log)
        log.show_task(None if log.task_id == message.task_id else message.task_id)
        self.query_one(Tasks).redraw()

    def action_cycle(self) -> None:
        if self._snapshot is None:
            return
        found = {history.group(node) for node in self._snapshot.compute.nodes}
        cycle: tuple[NodeGroup | None, ...] = (None, *(group for group in NODE_GROUPS if group in found))
        self._group = cycle[(cycle.index(self._group) + 1) % len(cycle)] if self._group in cycle else None
        self._paint()

    def action_group(self, name: str) -> None:
        for group in NODE_GROUPS:
            if group == name:
                self._group = None if self._group == group else group
        self._paint()

    def action_move(self, step: int) -> None:
        if self._snapshot is None:
            return
        keys = [node.id for node in _listed(self._snapshot.compute, self._group)]
        if not keys:
            return
        at = keys.index(self._selected) if self._selected in keys else 0
        self._selected = keys[max(0, min(len(keys) - 1, at + step))]
        self._paint()

    def action_expand(self, node_id: str | None = None) -> None:
        if (node := self._node(node_id)) is None:
            return
        self._selected = node.id
        self._expanded = self._expanded ^ {node.id}
        self._paint()

    def action_logs(self, node_id: str | None = None) -> None:
        if (node := self._node(node_id)) is not None:
            log = self.query_one(Log)
            log.show_node(None if log.node_id == node.id else node.id)
            self._paint()

    def action_kill(self, node_id: str | None = None) -> None:
        node = self._node(node_id)
        if node is None or node.terminated_at is not None or node.state not in history.DRAINABLE:
            return
        note = "it drains first, and a new node takes its rank if the compute still wants it"
        self.app.push_screen(
            Confirm(f"kill #{node.rank}?", "yes, kill", note=note),
            self._confirmed("kill", partial(writes.drain, self._client, self._compute, node.id)),
        )

    def action_scale(self) -> None:
        if self._snapshot is None or not history.deletable(self._snapshot.compute):
            return
        compute = self._snapshot.compute

        def scaled(bounds: tuple[int, int] | None) -> None:
            if bounds is not None:
                minimum, maximum = bounds
                self._perform("scale", partial(writes.scale, self._client, compute, minimum, maximum))

        self.app.push_screen(Scale(_minimum(compute.spec.nodes), _maximum(compute.spec.nodes)), scaled)

    def action_delete(self) -> None:
        if self._snapshot is None or not history.deletable(self._snapshot.compute):
            return
        self.app.push_screen(
            Confirm("delete this compute?", "yes, delete"),
            self._confirmed("delete", partial(writes.delete, self._client, self._compute)),
        )

    def _confirmed(self, action: str, write: Callable[[], Awaitable[None]]) -> Callable[[bool | None], None]:
        def answered(confirmed: bool | None) -> None:
            if confirmed:
                self._perform(action, write)

        return answered

    def _perform(self, action: str, write: Callable[[], Awaitable[None]]) -> None:
        self.run_worker(self._write(action, write), group="write")

    async def _write(self, action: str, write: Callable[[], Awaitable[None]]) -> None:
        try:
            await write()
        except FAILURES as error:
            self.notify(f"{action} failed: {_reason(error)}", severity="error")
            return
        self._read_now()

    def _poll(self) -> None:
        if self._reader is None or self._reader.is_finished:
            self._read_now()

    def _read_now(self) -> None:
        self._reader = self.run_worker(self._read(), group="snapshot", exclusive=True)

    async def _read(self) -> None:
        try:
            self._snapshot = await history.snapshot(self._client, self._compute, datetime.now(UTC))
        except FAILURES as error:
            self._failure = _reason(error)
        else:
            self._failure = None
        if not self.display:
            return
        if self._snapshot is not None:
            ranks = {node.id: node.rank for node in self._snapshot.compute.nodes}
            self.query_one(Tasks).set_ranks(ranks)
            self.query_one(Log).set_ranks(ranks)
        self._paint()

    def _spin(self) -> None:
        self._tick += 1
        snapshot = self._snapshot
        if snapshot is not None and (snapshot.compute.status.state in MOVING or any(node.state in NODE_MOVING for node in history.alive(snapshot.compute))):
            self._paint()

    def _paint(self) -> None:
        frame = _SPINNER_FRAMES[self._tick % len(_SPINNER_FRAMES)]
        snapshot = self._snapshot
        self.query_one("#actions", Static).update(_actions(snapshot))
        self.query_one("#header", Static).update(_header(snapshot, self._failure, frame))
        if snapshot is None:
            return
        self.query_one("#groups", Static).update(_title(snapshot.compute, self._group))
        self._paint_nodes(snapshot, frame)

    def _paint_nodes(self, snapshot: Snapshot, frame: str) -> None:
        listed = _listed(snapshot.compute, self._group)
        if self._selected not in {node.id for node in listed}:
            self._selected = listed[0].id if listed else None
        if not listed:
            self.query_one("#nodes", Static).update(Text("no nodes" if self._group is None else f"no {self._group} nodes", style=DIM))
            return
        logging = self.query_one(Log).node_id
        band = Style(bgcolor=self.app.theme_variables.get("panel"))
        nodes = self.query_one("#nodes", Static)
        cursor = self._selected if nodes.has_focus else None
        blocks = (_block(node, snapshot, frame, _Shown(node.id in self._expanded, node.id == cursor, node.id == logging), band) for node in listed)
        nodes.update(Text("\n").join(line for block in blocks for line in block))

    def _node(self, node_id: str | None) -> NodeResource | None:
        wanted = self._selected if node_id is None else node_id
        if self._snapshot is None or wanted is None:
            return None
        return next((node for node in self._snapshot.compute.nodes if node.id == wanted), None)


@dataclass(frozen=True, slots=True)
class _Shown:
    """How one node's block is drawn right now: open on its trends, under the cursor of a focused list, the one the log follows."""

    open: bool
    selected: bool
    logging: bool


_INDENT = "    "

_LIFECYCLE: tuple[NodeState, ...] = ("requested", "provisioning", "connecting", "bootstrapping", "ready", "draining", "lost", "deleting", "deleted", "failed")


def _reason(error: Exception) -> str:
    return str(error) or type(error).__name__


def _minimum(bounds: NodeBounds) -> int:
    return bounds.initial if bounds.min is None else bounds.min


def _maximum(bounds: NodeBounds) -> int:
    return bounds.initial if bounds.max is None else bounds.max


def _duration(start: datetime, end: datetime) -> str:
    return _format_duration((aware(end) - aware(start)).total_seconds())


def _ratio(used: float | None, total: float | None) -> str | None:
    return None if used is None or total is None else f"{used / 1024:.1f}/{total / 1024:.1f} GB"


def _speed(kbps: float) -> str:
    return f"{kbps / 1000:.1f}Mb/s" if kbps >= 1000 else f"{kbps:.0f}kb/s"


def _reading(node: NodeResource, name: str) -> float | None:
    metrics = node.metrics
    if isinstance(metrics, UnsetType) or name not in metrics:
        return None
    return metrics[name].value


def _actions(snapshot: Snapshot | None) -> Text:
    text = link("← computes", "app.pop_screen")
    if snapshot is not None and history.deletable(snapshot.compute):
        for label, action in (("scale", "scale"), ("delete", "delete")):
            text.append("   ")
            text.append_text(link(label, action))
    return text


def _header(snapshot: Snapshot | None, failure: str | None, frame: str) -> Text:
    lines = [Text("loading…", style=DIM)] if snapshot is None else _summary(snapshot, frame)
    if failure is not None:
        lines.append(Text(f"daemon request failed: {failure}", style=ERROR_STYLE))
    return Text("\n").join(lines)


def _summary(snapshot: Snapshot, frame: str) -> list[Text]:
    compute = snapshot.compute
    title = Text(compute.name or compute.id, style="bold")
    title.append("  ")
    title.append_text(state_badge(compute.status.state, frame))
    lines = [title, Text(_facts(snapshot), style=DIM)]
    if averages := _averages(snapshot.averages):
        lines.append(Text(f"avg {averages}", style=DIM))
    if idleness := _idleness(snapshot):
        lines.append(idleness)
    if error := compute.status.last_error:
        lines.append(Text(error.message, style=ERROR_STYLE))
    lines.append(Text(dots(compute.id, f"updated {clock(snapshot.at)}"), style=DIM))
    return lines


def _facts(snapshot: Snapshot) -> str:
    compute = snapshot.compute
    ended = compute.ended
    parts = (
        _population(compute),
        _machine(compute),
        f"cost ${money(ended.cost)}" if ended else f"${money(compute.cost)} spent",
        "" if ended else f"${money(compute.rate)}/h",
        _duration(compute.created_at, ended.at if ended else snapshot.at),
        f"nodes {_minimum(compute.spec.nodes)}–{_maximum(compute.spec.nodes)}",
    )
    return dots(*(part for part in parts if part))


def _population(compute: ComputeResource) -> str:
    counts = Counter(node.state for node in history.alive(compute))
    if not counts:
        return f"{sum(node.state == 'ready' for node in compute.nodes)}/{compute.spec.nodes.initial} nodes"
    return ", ".join(f"{counts[state]} {state}" for state in _LIFECYCLE if counts[state])


def _machine(compute: ComputeResource) -> str:
    spec = compute.spec.specs[0] if compute.spec.specs else None
    kind = compute.provider.kind if compute.provider else (spec.provider.kind if spec else "")
    name, count = _accelerator(compute)
    shape = f"{count}× {_accelerator_label(name)}" if name else ""
    return " ".join(part for part in (kind, shape) if part)


def _accelerator(compute: ComputeResource) -> tuple[str | None, int]:
    if compute.offer:
        return compute.offer.accelerator, compute.offer.accelerator_count
    if compute.spec.specs:
        return compute.spec.specs[0].accelerator, compute.spec.specs[0].accelerator_count
    return None, 1


def _averages(averages: Mapping[str, float]) -> str:
    parts = (
        f"cpu {averages['cpu']:.0f}%" if "cpu" in averages else "",
        f"gpu {averages['gpu_util']:.0f}%" if "gpu_util" in averages else "",
        f"mem {memory}" if (memory := _ratio(averages.get("mem_used_mb"), averages.get("mem_total_mb"))) else "",
        f"disk {averages['disk_used_pct']:.0f}%" if "disk_used_pct" in averages else "",
    )
    return "  ".join(part for part in parts if part)


def _idleness(snapshot: Snapshot) -> Text | None:
    compute = snapshot.compute
    running = compute.tasks.running
    idle = snapshot.idle
    if compute.ended or (idle is None and (running > 0 or "gpu_util" in snapshot.averages)):
        return None
    tasks = f"{running} running" if running else "no task"
    if idle is None:
        return Text(f"idle · {tasks}", style=WARNING_STYLE)
    quiet = f"{'≥' if idle.capped else ''}{_duration(idle.since, snapshot.at)}"
    if running:
        return Text(f"gpu idle {quiet} · {tasks}", style=ERROR_STYLE)
    return Text(f"idle {quiet} · {tasks}", style=WARNING_STYLE)


def _title(compute: ComputeResource, selected: NodeGroup | None) -> Text:
    counts = Counter(history.group(node) for node in compute.nodes)
    if not counts:
        return Text("Nodes (0)", style="bold")
    text = Text("Nodes (", style="bold")
    for index, group in enumerate(group for group in NODE_GROUPS if counts[group]):
        if index:
            text.append(" · ", style="bold")
        label = f"{counts[group]} {group}{' ✕' if selected == group else ''}"
        text.append_text(link(label, f"group('{group}')", "bright_black" if selected not in (None, group) else "bold"))
    text.append(")", style="bold")
    return text


def _listed(compute: ComputeResource, group: NodeGroup | None) -> tuple[NodeResource, ...]:
    shown = (node for node in compute.nodes if group is None or history.group(node) == group)
    return tuple(sorted(shown, key=lambda node: (node.terminated_at is not None, node.rank, node.created_at)))


def _block(node: NodeResource, snapshot: Snapshot, frame: str, shown: _Shown, band: Style) -> list[Text]:
    terminated = node.terminated_at is not None
    lines = [_head(node, snapshot.at, frame, shown)]
    if shown.selected:
        lines[0].stylize(band)
    if node.last_error is not None:
        lines.append(Text(f"{_INDENT}{node.last_error.message}", style=ERROR_STYLE))
    if not terminated:
        lines.extend(line for line in (_load(node), *((_cpu_line(snapshot, node), _gpu_line(snapshot, node)) if shown.open else ())) if line is not None)
    if shown.open:
        parts = (node.address, node.id, f"machine {node.machine}" if node.machine else None)
        lines.append(Text(_INDENT + dots(*(part for part in parts if part)), style=DIM))
    return lines


def _head(node: NodeResource, now: datetime, frame: str, shown: _Shown) -> Text:
    terminated = node.terminated_at is not None
    text = Text()
    text.append_text(link("▾" if shown.open else "▸", f"expand('{node.id}')"))
    text.append(" ")
    text.append_text(rank_badge(node.rank, muted=terminated))
    text.append(" is ", style=DIM)
    text.append_text(_state(node, frame))
    if tail := _tail(node, now):
        text.append(f" {tail}", style=DIM)
    if not terminated and not isinstance(node.running, UnsetType) and node.running:
        text.append("  ")
        text.append_text(_doing(node, now))
    text.append("  ")
    text.append_text(link("logs ✕" if shown.logging else "logs", f"logs('{node.id}')"))
    if not terminated and node.state in history.DRAINABLE:
        text.append("  ")
        text.append_text(link("kill", f"kill('{node.id}')"))
    return text


def _tail(node: NodeResource, now: datetime) -> str:
    parts = (
        _life(node, now),
        "" if node.price_per_hour is None else f"@ ${money(node.price_per_hour)}/h",
        f"({node.market.replace('_', ' ')})" if node.market else "",
    )
    return " ".join(part for part in parts if part)


def _life(node: NodeResource, now: datetime) -> str:
    if node.launched_at is None:
        return ""
    if node.terminated_at is None:
        return f"up {_duration(node.launched_at, now)}"
    return f"ran {_duration(node.launched_at, node.terminated_at)}"


def _load(node: NodeResource) -> Text | None:
    cpu, gpu, disk = _reading(node, "cpu"), _reading(node, "gpu_util"), _reading(node, "disk_used_pct")
    parts = (
        ("cpu", None if cpu is None else f"{cpu:.0f}%"),
        ("gpu", None if gpu is None else f"{gpu:.0f}%"),
        ("memory", _ratio(_reading(node, "mem_used_mb"), _reading(node, "mem_total_mb"))),
        ("disk", None if disk is None else f"{disk:.0f}%"),
    )
    known = [(label, value) for label, value in parts if value is not None]
    if not known:
        return None
    text = Text(_INDENT)
    for index, (label, value) in enumerate(known):
        if index:
            text.append("  ")
        text.append(f"{label} ", style=DIM)
        text.append(value)
    return text


def _state(node: NodeResource, frame: str) -> Text:
    word: str = node.state
    phases = () if isinstance(node.phases, UnsetType) else node.phases
    if node.state == "bootstrapping" and (active := next((phase.name for phase in reversed(phases) if phase.state == "started"), None)):
        word = _PHASE_LABELS.get(active, active)
    if node.state in NODE_MOVING:
        word = f"{frame} {word}"
    text = Text(word, style=DIM if node.terminated_at is not None else NODE_STYLES.get(node.state, ""))
    if node.address is None and node.progress is not None:
        completion = f" {node.progress.completion * 100:.0f}%" if node.progress.completion is not None else ""
        text.append(f" {node.progress.step}{completion}", style=DIM)
    return text


def _doing(node: NodeResource, now: datetime) -> Text:
    running = () if isinstance(node.running, UnsetType) else node.running
    if not running:
        return Text("idle", style=DIM)
    oldest = sorted(running, key=lambda task: aware(task.started_at or now))
    text = Text("▶ ", style="green")
    if len(oldest) > NAMED_TASKS:
        text.append(str(len(oldest)))
        text.append(f" {_age(oldest[0].started_at, now)}", style=DIM)
        return text
    for index, task in enumerate(oldest):
        if index:
            text.append(" · ", style=DIM)
        text.append(task.function.name or "task")
        text.append(f" {_age(task.started_at, now)}", style=DIM)
    return text


def _age(started_at: datetime | None, now: datetime) -> str:
    return "" if started_at is None else _duration(started_at, now)


def _cpu_line(snapshot: Snapshot, node: NodeResource) -> Text | None:
    if _reading(node, "cpu") is None:
        return None
    received, sent = _reading(node, "net_rx_kbps"), _reading(node, "net_tx_kbps")
    details = f"net ↓{_speed(received)} ↑{_speed(sent)}" if received is not None and sent is not None else ""
    return _trend("cpu", snapshot.sparks.get((node.id, "cpu"), ()), details)


def _gpu_line(snapshot: Snapshot, node: NodeResource) -> Text | None:
    if _reading(node, "gpu_util") is None:
        return None
    memory = _ratio(_reading(node, "gpu_mem_mb"), _reading(node, "gpu_mem_total_mb"))
    temperature, power = _reading(node, "gpu_temp_c"), _reading(node, "gpu_power_w")
    parts = (
        f"vram {memory}" if memory else "",
        f"temp {temperature:.0f}°C" if temperature is not None else "",
        f"power {power:.0f} W" if power is not None else "",
    )
    return _trend("gpu", snapshot.sparks.get((node.id, "gpu_util"), ()), dots(*(part for part in parts if part)))


def _trend(label: str, values: Sequence[float | None], details: str) -> Text:
    text = Text(_INDENT)
    text.append(f"{label} ", style=DIM)
    text.append(spark(tuple(dropwhile(lambda value: value is None, values))))
    if details:
        text.append(f"  {details}")
    return text


__all__ = ["ComputeScreen"]
