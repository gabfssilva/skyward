"""``sky app``: the fleet on one screen, and one compute under the cursor.

A Textual application over :class:`~skyward.core.fleet.FleetObserver`. The
observer folds; this module only draws, on a timer rather than per event — a
compute streaming gauges from eight machines would otherwise repaint the table
faster than a terminal can show it, and the timer is also what moves the spinner
and the clock. Every cell is a Rich renderable drawn with the console's own
vocabulary (:mod:`skyward.core.widgets`), so the table and the footer a pool
prints are the same words in the same colours.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.theme import Theme
from textual.widgets import DataTable, Footer, Static

from skyward.core.client import Client, address
from skyward.core.fleet import Fleet, FleetObserver
from skyward.core.view import ComputeView, NodeView, TaskView
from skyward.core.widgets import (
    _DARK,
    _PHASE_LABELS,
    _SPINNER_FRAMES,
    DIM,
    WARNING_STYLE,
    _accelerator_label,
    _badge_style,
    _fill_style,
    _format_duration,
    _inline_badge,
)

REFRESH = 0.25
"""Seconds between repaints — the spinner's pace, and the most a gauge is redrawn."""

NAMED_TASKS = 3
"""Up to this many tasks on one node are named in its row; past it, the row says how many."""

LIGHT = Theme(
    name="skyward-light",
    primary="#1f6b3a",
    secondary="#2f5a8a",
    warning="#b45309",
    error="#b91c1c",
    success="#1f6b3a",
    accent="#2f5a8a",
    foreground="#111111",
    background="#ffffff",
    surface="#ffffff",
    panel="#f2f2f2",
    dark=False,
    variables={"footer-background": "#f2f2f2", "footer-key-foreground": "#1f6b3a"},
)
"""A white page: the terminal's own light background is the theme, not a tint of it."""

COLUMNS = ("NAME", "STATE", "PROVIDER", "ACCELERATOR", "NODES", "RATE", "UP", "TASKS")
NODE_COLUMNS = ("NODE", "STATE", "MACHINE", "MKT", "ADDRESS", "GPU", "MEM", "TASK")


class Dashboard(App[None]):
    """The screen ``sky app`` opens: every live compute, and the one under the cursor."""

    TITLE = "skyward"
    CSS = """
    Screen { layout: vertical; }
    #summary { height: 1; margin: 1 2 1 2; }
    #computes { height: auto; max-height: 40%; margin: 0 1; background: transparent; }
    Detail { height: 1fr; margin: 1 1 0 1; }
    Detail > #nodes { height: auto; background: transparent; }
    Detail > #tail { margin-top: 1; height: 1fr; }
    DataTable > .datatable--header { color: $text-muted; background: transparent; text-style: none; }
    DataTable > .datatable--cursor { background: $panel; }
    """
    BINDINGS = [Binding("q", "quit", "quit")]
    ENABLE_COMMAND_PALETTE = False

    def __init__(self, client: Client, url: str) -> None:
        super().__init__()
        self.register_theme(LIGHT)
        self.theme = "textual-dark" if _DARK else LIGHT.name
        self._fleet = FleetObserver(client)
        self._url = url

    def on_mount(self) -> None:
        self.run_worker(self._fleet.follow(), exclusive=True)
        self.push_screen(FleetScreen(self._fleet, self._url))


class FleetScreen(Screen[None]):
    """The table of live computes, and the detail of the one under the cursor."""

    BINDINGS = [Binding("enter", "open", "open compute", priority=True)]

    def __init__(self, fleet: FleetObserver, url: str) -> None:
        super().__init__()
        self._fleet = fleet
        self._url = url
        self._selected: str | None = None
        self._tick = 0

    def compose(self) -> ComposeResult:
        yield Static(id="summary")
        table = DataTable[Text](id="computes", cursor_type="row", zebra_stripes=False)
        table.add_columns(*COLUMNS)
        yield table
        yield Detail()
        yield Footer()

    def on_mount(self) -> None:
        self.set_interval(REFRESH, self._repaint)
        self._repaint()

    def on_data_table_row_highlighted(self, message: DataTable.RowHighlighted) -> None:
        self._selected = message.row_key.value

    def action_open(self) -> None:
        if self._selected is not None:
            self.app.push_screen(ComputeScreen(self._fleet, self._selected))

    def _repaint(self) -> None:
        self._tick += 1
        frame = _SPINNER_FRAMES[self._tick % len(_SPINNER_FRAMES)]
        fleet = self._fleet.views
        now = datetime.now(UTC)
        self.query_one("#summary", Static).update(summary(fleet, self._url, frame))
        table = self.query_one("#computes", DataTable)
        ordered = sorted(fleet.values(), key=rate, reverse=True)
        hovered = table.hover_coordinate.row
        errors = ordered[hovered].errors if 0 <= hovered < len(ordered) else ()
        table.tooltip = Text("\n".join(errors), style=WARNING_STYLE) if errors else None
        table.clear()
        for view in ordered:
            table.add_row(*row(view, now, frame), key=view.id)
        if self._selected not in fleet:
            self._selected = ordered[0].id if ordered else None
        if self._selected is not None:
            table.move_cursor(row=[view.id for view in ordered].index(self._selected), animate=False)
        self.query_one(Detail).show(fleet.get(self._selected or ""), now, frame)


class ComputeScreen(Screen[None]):
    """One compute, full height: the same detail, with room for its output."""

    BINDINGS = [Binding("escape", "app.pop_screen", "back")]

    def __init__(self, fleet: FleetObserver, compute_id: str) -> None:
        super().__init__()
        self._fleet = fleet
        self._compute = compute_id
        self._tick = 0

    def compose(self) -> ComposeResult:
        yield Detail()
        yield Footer()

    def on_mount(self) -> None:
        self.set_interval(REFRESH, self._repaint)
        self._repaint()

    def _repaint(self) -> None:
        self._tick += 1
        frame = _SPINNER_FRAMES[self._tick % len(_SPINNER_FRAMES)]
        view = self._fleet.views.get(self._compute)
        if view is None:
            self.app.pop_screen()
            return
        self.query_one(Detail).show(view, datetime.now(UTC), frame)


class Detail(Vertical):
    """The compute under the cursor: its nodes, and what they last printed, down to the bottom of the screen."""

    def compose(self) -> ComposeResult:
        table = DataTable[Text](id="nodes", cursor_type="none")
        table.add_columns(*NODE_COLUMNS)
        yield table
        yield Static(id="tail")

    def show(self, view: ComputeView | None, now: datetime, frame: str) -> None:
        table = self.query_one("#nodes", DataTable)
        table.clear()
        if view is None:
            self.query_one("#tail", Static).update("")
            return
        running: dict[str, list[TaskView]] = {}
        for task in view.tasks:
            if task.state == "running" and task.node:
                running.setdefault(task.node, []).append(task)
        for node in sorted(view.nodes, key=lambda node: node.rank):
            table.add_row(*node_row(node, tuple(running.get(node.id, ())), now, frame), key=node.id)
        output = self.query_one("#tail", Static)
        output.update(tail(view, max(1, output.content_size.height)))


def summary(fleet: Fleet, url: str, frame: str) -> Text:
    """The line above the table: the daemon, and the fleet's totals."""
    nodes = [node for view in fleet.values() for node in view.nodes if node.state != "deleted"]
    tasks = Counter(task.state for view in fleet.values() for task in view.tasks)
    text = Text()
    text.append(f" {frame} skyward ", style=_badge_style("skyward"))
    text.append(f"  {where(url)}   ", style=DIM)
    text.append(
        _dots(
            f"{len(fleet)} computes",
            f"{len(nodes)} nodes",
            f"{sum(node.state == 'ready' for node in nodes)} ready",
        ),
    )
    text.append(" · ", style=DIM)
    text.append(f"${sum(map(rate, fleet.values())):.2f}/hr", style="green bold")
    if tasks["running"] or tasks["queued"]:
        text.append(" · ", style=DIM)
        text.append(_dots(*(f"{tasks[state]} {word}" for state, word in (("running", "tasks running"), ("queued", "queued")) if tasks[state])))
    return text


def where(url: str) -> str:
    """The daemon, as a person would say it: ``@ :17590`` on this machine, the url anywhere else."""
    host, port = address(url)
    return f"@ :{port}" if host in _LOCAL else url


def row(view: ComputeView, now: datetime, frame: str) -> tuple[Text, ...]:
    """One compute, as the table's cells."""
    name = Text(view.name or view.id, style="bold")
    if view.errors:
        name.append(" ⚠", style=WARNING_STYLE)
    return (
        name,
        state_badge(view.state, frame),
        provider(view),
        Text(accelerator(view)),
        nodes(view),
        Text(f"${rate(view):.2f}/hr", style="green" if rate(view) else DIM, justify="right"),
        Text(uptime(view, now), justify="right"),
        tasks(view),
    )


def node_row(node: NodeView, running: tuple[TaskView, ...], now: datetime, frame: str) -> tuple[Text, ...]:
    """One machine of the compute under the cursor, as the detail's cells."""
    gpu = _last(node, "gpu_util")
    memory = _last(node, "gpu_mem_mb") or _last(node, "mem_used_mb")
    return (
        Text(str(node.rank)),
        node_state(node, frame),
        Text(_short(node.machine), style=DIM),
        Text(node.market or ""),
        Text(node.address or "", style=DIM),
        _gauge(gpu),
        Text(f"{memory / 1024:.0f} GB" if memory else ""),
        doing(running, now),
    )


def doing(running: tuple[TaskView, ...], now: datetime) -> Text:
    """What a node is running: each task by name up to ``NAMED_TASKS``, and past that the count and the oldest's age."""
    if not running:
        return Text("idle", style=DIM)
    oldest = sorted(running, key=lambda task: task.started_at or now)
    text = Text("▶ ", style="green")
    if len(oldest) > NAMED_TASKS:
        text.append(str(len(oldest)))
        text.append(f" {_age(oldest[0], now)}", style=DIM)
        return text
    for index, task in enumerate(oldest):
        if index:
            text.append(" · ", style=DIM)
        text.append(task.function or "task")
        text.append(f" {_age(task, now)}", style=DIM)
    return text


def tail(view: ComputeView, lines: int) -> Text:
    """The last lines the nodes printed, newest node output last, at most ``lines``."""
    text = Text()
    spoken = [(node.rank, line) for node in sorted(view.nodes, key=lambda node: node.rank) for line in node.tail[-lines:]]
    for index, (rank, line) in enumerate(spoken[-lines:]):
        if index:
            text.append("\n")
        text.append(f"[node {rank}] ", style=DIM)
        text.append(line)
    return text


def state_badge(state: str, frame: str) -> Text:
    label = f"{frame} {state}" if state in _MOVING else state
    return Text(f" {label} ", style=_badge_style(_BADGE_KEYS.get(state, state)))


def node_state(node: NodeView, frame: str) -> Text:
    word = node.state
    if node.state == "bootstrapping" and (active := next((phase.name for phase in reversed(node.phases) if phase.started and not phase.finished), None)):
        word = _PHASE_LABELS.get(active, active)
    if node.state in _NODE_MOVING:
        word = f"{frame} {word}"
    return Text(word, style=_NODE_STYLES.get(node.state, ""))


def provider(view: ComputeView) -> Text:
    text = _inline_badge(view.provider.upper()) if view.provider else Text()
    if view.region:
        text.append(f" {view.region}", style=DIM)
    return text


def accelerator(view: ComputeView) -> str:
    if view.accelerator:
        return f"{view.accelerator_count}× {_accelerator_label(view.accelerator)}"
    return _dots(*(part for part in (f"{view.cpus} vCPU" if view.cpus else "", f"{view.memory_gb:g} GB" if view.memory_gb else "") if part))


def nodes(view: ComputeView) -> Text:
    text = Text()
    for node in sorted(view.nodes, key=lambda node: node.rank):
        text.append("■", style=_NODE_STYLES.get(node.state, ""))
    text.append(f" {view.nodes_ready}/{view.nodes_total or len(view.nodes)}", style=DIM)
    return text


def tasks(view: ComputeView) -> Text:
    counts = Counter(task.state for task in view.tasks)
    failed = counts["failed"] + counts["timed_out"] + counts["indeterminate"] + counts["cancelled"]
    parts: list[Text] = []
    if counts["running"]:
        parts.append(Text(f"▶ {counts['running']}", style="green"))
    if counts["queued"]:
        parts.append(Text(f"{counts['queued']} queued"))
    if counts["succeeded"]:
        parts.append(Text(f"✓ {counts['succeeded']}"))
    if failed:
        parts.append(Text(f"✗ {failed}", style="red"))
    if not parts:
        return Text("idle", style=DIM)
    if not counts["running"] and not counts["queued"]:
        parts.insert(0, Text("idle", style=DIM))
    return Text(" · ", style=DIM).join(parts)


def rate(view: ComputeView) -> float:
    """What the compute costs per hour right now: the price of every machine still held."""
    return sum(node.price_per_hour or 0.0 for node in view.nodes if node.state in _BILLING)


def uptime(view: ComputeView, now: datetime) -> str:
    return _format_duration((now - _aware(view.created_at)).total_seconds()) if view.created_at else ""


_LOCAL = frozenset({"127.0.0.1", "localhost", "::1"})
_MOVING = frozenset({"requested", "provisioning", "deleting"})
_NODE_MOVING = frozenset({"requested", "provisioning", "connecting", "bootstrapping", "draining", "deleting"})
_BILLING = frozenset({"provisioning", "connecting", "bootstrapping", "ready", "draining", "deleting"})
_BADGE_KEYS = {"degraded": "failed", "deleting": "shutting down", "requested": "queued"}
_NODE_STYLES = {
    "requested": "bright_black",
    "provisioning": "bright_black",
    "connecting": "yellow",
    "bootstrapping": "yellow",
    "ready": "green",
    "draining": "yellow",
    "lost": "red",
    "failed": "red",
    "deleting": "color(238)",
    "deleted": "color(238)",
}


def _dots(*parts: str) -> str:
    return " · ".join(parts)


def _age(task: TaskView, now: datetime) -> str:
    return _format_duration((now - _aware(task.started_at)).total_seconds()) if task.started_at else ""


def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def _short(machine: str | None) -> str:
    if machine is None:
        return ""
    return machine if len(machine) <= 12 else f"{machine[:5]}…{machine[-4:]}"


def _last(node: NodeView, name: str) -> float | None:
    samples = node.metrics.get(name)
    return samples[-1] if samples else None


def _gauge(percentage: float | None) -> Text:
    """Eight cells of load, filling from the colour of `provisioning` to the colour of `ready`."""
    if percentage is None:
        return Text("")
    done = min(max(percentage, 0.0), 100.0) / 100
    filled = round(done * _GAUGE_WIDTH)
    text = Text()
    text.append("━" * filled, style=_fill_style(45 + 75 * done))
    text.append("━" * (_GAUGE_WIDTH - filled), style=DIM)
    text.append(f" {percentage:.0f}%", style="" if percentage >= 5 else DIM)
    return text


_GAUGE_WIDTH = 8


__all__ = ["Dashboard", "row", "summary", "tail"]
