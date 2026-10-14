"""The live console: the compute as a panel at the foot of the terminal, and its log above it.

The panel is redrawn every frame and says where the compute is now. A header
names it — what it is made of, where, what it costs — and under it comes one line
per setup stage under way, ``provision`` and ``bootstrap``, each with its own
timer, and once the compute has been ready, its tasks and the gauges across its
nodes. A node that goes away while the compute runs brings the setup stages back
above the tasks until its replacement is ready, so recovering and running are
drawn as one picture.

A stage lists the nodes it is waiting on, one line each, up to :data:`LISTED`.
Past that it groups them by the step they are on and names only the slowest, so a
hundred machines coming up take as many lines as there are steps.

Everything else is scrollback: what the nodes print, behind their rank, and a
line for what happened to a node (lost, replaced, joined, refreshed) or to a task
(failed, timed out, retried). What the view does not keep, the console remembers
in :class:`Seen`: when it saw each node begin each stage, which ranks went away,
and what the gauges read over the whole run, for the summary.
"""

from __future__ import annotations

import logging
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field, replace
from itertools import groupby
from statistics import fmean
from types import MappingProxyType
from typing import Literal, TextIO

from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.live import Live
from rich.text import Text

from skyward.api.v1 import TaskCounts
from skyward.core.console import Watcher
from skyward.core.view import ComputeView, NodeView, TaskView
from skyward.core.widgets import _PHASE_LABELS, _SPINNER_FRAMES, _accelerator_label, _bar
from skyward.shared.events import ComputeDegraded, ComputeDeleted, ConsoleEvent, Event, MetricEvent, NodeEvent, TaskEvent
from skyward.shared.observability import NAME as LOGGER_NAME
from skyward.shared.schemas import NodeState

WIDTH = 100
"""The widest the panel is drawn: on a wider terminal the timers would sit too far from what they time."""

LISTED = 4
"""The most nodes a stage lists one line each; past it, they are grouped by the step they are on."""

type Reason = Literal["first", "replacing", "adding", "returning"]
"""Why a node is in a setup: it is the compute's first, it holds a rank that went away, it is one rank more, or it is refreshed in place."""


@dataclass(frozen=True, slots=True)
class Times:
    """When the console saw one node begin each stage of its setup, in seconds since the epoch."""

    reason: Reason = "first"
    provision: float | None = None
    bootstrap: float | None = None
    ready: float | None = None


@dataclass(frozen=True, slots=True)
class Spans:
    """How long the stages of a setup took, in seconds, where the console saw both ends of one."""

    provision: float | None = None
    bootstrap: float | None = None


@dataclass(frozen=True, slots=True)
class Fleet:
    """What the compute is made of, as the header says it — kept, because by the summary its machines are gone."""

    nodes: int = 0
    market: str = ""
    memory_gb: float | None = None


@dataclass(frozen=True, slots=True)
class Seen:
    """What the console remembers that the view does not keep."""

    times: Mapping[str, Times] = field(default_factory=lambda: MappingProxyType({}))
    setup: frozenset[str] = frozenset()
    """The nodes of the setup under way: those still coming up, and those that came up since it began."""
    opening: bool = False
    """Whether the setup under way began before the compute was ever ready."""
    first: Spans | None = None
    """How long the compute's first setup took, once it is over."""
    ran: bool = False
    lost: frozenset[int] = frozenset()
    """The ranks whose node went away and whose replacement is not ready yet."""
    fleet: Fleet = Fleet()
    usage: Mapping[str, Mapping[str, tuple[float, int]]] = field(default_factory=lambda: MappingProxyType({}))
    """Per gauge, per node: the sum of its readings and how many there were."""


class RichConsole(Watcher):
    """The panel and the scrollback, fed by the pool's :class:`~skyward.core.console.Observer` like any other watcher."""

    def __init__(self, out: TextIO | None = None) -> None:
        self._console = Console(file=out or sys.stderr)
        self._seen = Seen()
        self._panel = _Panel()
        self._live: Live | None = None
        self._stack = ExitStack()

    def opened(self, view: ComputeView) -> None:
        self._show(view, advance(self._seen, view, time.time()))
        self._stack.enter_context(_quiet(self._console))
        self._live = Live(self._panel, console=self._console, refresh_per_second=8, transient=True, redirect_stdout=True, redirect_stderr=False)
        self._live.start()

    def event(self, event: Event, view: ComputeView) -> None:
        if self._live is None:
            return
        seen = advance(self._seen, view, time.time())
        if line := said(event, view, seen):
            self._console.print(line)
        self._show(view, heard(seen, event, view))
        match event:
            case ComputeDeleted():
                self._end(summary(view, self._seen, time.time(), self._width))
            case _:
                pass

    def refreshed(self, view: ComputeView) -> None:
        if self._live is not None:
            self._show(view, advance(self._seen, view, time.time()))

    def closed(self, view: ComputeView) -> None:
        if self._live is not None:
            match view.state:
                case "deleting" | "deleted":
                    self._end(summary(view, self._seen, time.time(), self._width))
                case _:
                    self._end((detached(view, time.time()),))
        self._stack.close()

    @property
    def _width(self) -> int:
        return min(self._console.width, WIDTH)

    def _show(self, view: ComputeView, seen: Seen) -> None:
        self._seen = seen
        self._panel.state = (view, seen)

    def _end(self, lines: Sequence[Text]) -> None:
        """Take the panel down and leave ``lines`` where it was."""
        if self._live is not None:
            self._live.stop()
            self._live = None
        for line in lines:
            self._console.print(line)


def advance(seen: Seen, view: ComputeView, now: float) -> Seen:
    """What the console remembers once it has looked at the view: each node's stages, the setup under way, the fleet."""

    def timed(node: NodeView) -> Times:
        times = seen.times.get(node.id)
        if node.state in _SETUP and (times is None or times.ready is not None):
            times = Times(reason=reason(node, times))
        if times is None:
            return Times(ready=now) if node.state in _UP else Times()
        if node.state in _PROVISION and times.provision is None:
            return replace(times, provision=node.created_at.timestamp() if node.created_at else now)
        if node.state in _BOOTSTRAP and times.bootstrap is None:
            return replace(times, bootstrap=now)
        if node.state in _UP and times.ready is None:
            return replace(times, ready=now)
        return times

    def reason(node: NodeView, times: Times | None) -> Reason:
        if times is not None:
            return "returning"
        if not seen.ran:
            return "first"
        replaces = node.rank in seen.lost or any(other.rank == node.rank and other.state in _GONE for other in view.nodes)
        return "replacing" if replaces else "adding"

    setting = frozenset(node.id for node in view.nodes if node.state in _SETUP)
    setup = seen.setup | setting if setting else frozenset()
    kept = {node.id for node in view.nodes} | setup | seen.setup
    times = MappingProxyType({
        **{key: value for key, value in seen.times.items() if key in kept},
        **{node.id: timed(node) for node in view.nodes},
    })
    over = bool(seen.setup) and not setting
    return replace(
        seen,
        times=times,
        setup=setup,
        opening=(seen.opening if seen.setup else not seen.ran) if setting else False,
        first=_spans(seen.setup, times) if over and seen.opening and seen.first is None else seen.first,
        ran=seen.ran or view.state == "ready" or bool(view.tasks),
        fleet=_fleet(view, seen.fleet),
    )


def heard(seen: Seen, event: Event, view: ComputeView) -> Seen:
    """What the console remembers once it has heard the event: the ranks that went away, and the gauges."""
    match event:
        case NodeEvent(node=node, state="lost" | "failed") if (row := _row(view, node)) is not None:
            heirs = {
                other.id: replace(times, reason="replacing")
                for other in view.nodes
                if other.rank == row.rank and (times := seen.times.get(other.id)) is not None and times.reason == "adding"
            }
            return replace(seen, lost=seen.lost | {row.rank}, times=MappingProxyType({**seen.times, **heirs}))
        case NodeEvent(node=node, state="ready") if (row := _row(view, node)) is not None:
            return replace(seen, lost=seen.lost - {row.rank})
        case MetricEvent(node=node, name=name, value=value) if name in _GAUGES:
            readings = seen.usage.get(name, MappingProxyType({}))
            total, count = readings.get(node, (0.0, 0))
            usage = {**seen.usage, name: MappingProxyType({**readings, node: (total + value, count + 1)})}
            return replace(seen, usage=MappingProxyType(usage))
        case _:
            return seen


def said(event: Event, view: ComputeView, seen: Seen) -> Text | None:
    """The line an event leaves in the scrollback, if it leaves one.

    What a node prints while it is coming up is the panel's, not the log's: it is
    the line under the node's step, and it is replaced by the next one. A node that
    comes up before the compute was ever ready leaves nothing either — the panel
    said it, and the next thing it says is ``run``.
    """
    match event:
        case ConsoleEvent(node=node, content=content):
            row = _row(view, node)
            if row is not None and row.state in _SETUP:
                return None
            return _from(view, row, Text(content))
        case NodeEvent(node=node, state="lost" | "failed" as state, error=error):
            return _from(view, _row(view, node), _dotted(Text(f"✗ {state}", style="red"), _dim(error)))
        case NodeEvent(node=node, state="ready") if seen.ran and (row := _row(view, node)) is not None:
            times = seen.times.get(row.id, Times())
            word = "refreshed" if times.reason == "returning" else "replaced" if times.reason == "replacing" or row.rank in seen.lost else "joined"
            took = _spans(frozenset({row.id}), seen.times)
            return _from(view, row, _dotted(Text(f"✓ {word}", style="green"), *_took(took)))
        case TaskEvent(task=task, state="failed" | "timed_out" as state):
            row = _task(view, task)
            what = Text(f"✗ {_function(row)} {'failed' if state == 'failed' else 'timed out'}", style="red")
            return _from(view, _row(view, row.node) if row and row.node else None, what)
        case TaskEvent(task=task, state="retrying", attempt=attempt):
            row = _task(view, task)
            what = _dotted(Text(f"↻ {_function(row)} retrying", style="yellow"), _dim(f"attempt {attempt}"))
            return _from(view, _row(view, row.node) if row and row.node else None, what)
        case ComputeDegraded(error=error):
            return _dotted(Text("✗ degraded", style="red"), _dim(error))
        case _:
            return None


def panel(view: ComputeView, seen: Seen, now: float, width: int = WIDTH) -> tuple[Text, ...]:
    """The live panel, line by line: the header, then each stage under way, with what it is waiting on."""
    return (
        _header(view, seen, now, width),
        *_provision(view, seen, now, width),
        *_bootstrap(view, seen, now, width),
        *_run(view, seen, now, width),
    )


def summary(view: ComputeView, seen: Seen, now: float, width: int = WIDTH) -> tuple[Text, ...]:
    """What the panel becomes once the compute is gone: what it was, what it ran, how long it took to come up, how hard it worked."""
    elapsed = _span(now - view.created_at.timestamp()) if view.created_at else None
    head = _justified(Text.assemble(("✓ ", "green"), _title(view, seen)), _dotted(Text(elapsed or ""), _dim(_dollars(view.cost))), width)
    nodes = sorted({node for readings in seen.usage.values() for node in readings})
    averages = tuple(
        MappingProxyType({name: total / count for name, readings in seen.usage.items() if node in readings for total, count in (readings[node],)})
        for node in nodes
    )
    setup = _took(seen.first)
    rows = (
        ("tasks", _tally(view.task_counts or _counted(view.tasks))),
        ("setup", _dotted(*setup) if setup else None),
        ("usage", _gauges(averages)),
    )
    return (head, *(Text.assemble((f"  {name:<{_LABEL}}", "bright_black"), value) for name, value in rows if value is not None))


def detached(view: ComputeView, now: float) -> Text:
    """The line left when the console stops watching a compute that goes on without it."""
    up = f"up {_span(now - view.created_at.timestamp())}" if view.created_at else None
    return _dotted(Text("↩ detached", style="yellow"), _dim(f"{view.name or view.id} keeps running"), _dim(up), _dim(_dollars(view.cost)))


def render(view: ComputeView, now: float | None = None) -> RenderableType:
    """The panel for a view, as a console that has only ever seen this view would draw it."""
    moment = time.time() if now is None else now
    return Group(*panel(view, advance(Seen(), view, moment), moment))


_PROVISION: frozenset[NodeState] = frozenset({"requested", "provisioning"})
_BOOTSTRAP: frozenset[NodeState] = frozenset({"connecting", "bootstrapping"})
_SETUP = _PROVISION | _BOOTSTRAP
_UP: frozenset[NodeState] = frozenset({"ready", "draining"})
_LIVE = _SETUP | _UP
_GONE: frozenset[NodeState] = frozenset({"lost", "failed", "deleting", "deleted"})

_GAUGES = frozenset({"gpu_util", "gpu_mem_mb", "gpu_mem_total_mb", "cpu", "mem", "mem_used_mb", "mem_total_mb"})

_LABEL = 11
"""The width of a stage's name, so that every stage's detail starts on the same column."""

_HUES = ("cyan", "magenta", "yellow", "blue", "green", "bright_cyan", "bright_magenta", "bright_yellow", "bright_blue", "bright_green")
"""One colour per rank, so that the lines of a broadcast stay told apart."""

_STEPS = {label: order for order, label in enumerate(_PHASE_LABELS.values())}

_DOT = (" · ", "bright_black")


class _Panel:
    """What the live display redraws: the latest view and memory, drawn at the moment of the frame, so the timers move between events."""

    def __init__(self) -> None:
        self.state = (ComputeView(id=""), Seen())

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        view, seen = self.state
        yield from panel(view, seen, time.time(), min(options.max_width, WIDTH))


def _header(view: ComputeView, seen: Seen, now: float, width: int) -> Text:
    live = _live(view)
    rate = view.rate if view.rate else sum(node.price_per_hour or 0.0 for node in live)
    left = _dotted(_title(view, seen), _dim(f"${rate:.2f}/h" if rate else None))
    elapsed = _clock(now - view.created_at.timestamp()) if view.created_at else None
    return _justified(left, _dotted(Text(elapsed or ""), _dim(_dollars(view.cost))), width)


def _provision(view: ComputeView, seen: Seen, now: float, width: int) -> tuple[Text, ...]:
    live = _live(view)
    waiting = tuple(node for node in live if node.state in _PROVISION)
    members = _members(view, seen)
    starts = [times.provision for times in _times(seen, members) if times.provision is not None]
    starting = not live and not seen.ran and view.state in {"requested", "provisioning"}
    if not (waiting or starting or (seen.setup and starts)):
        return ()
    detail = Text.assemble(f"{len(live) - len(waiting)}/{len(live) or view.initial} machines", _note(members, seen, under_way=bool(waiting or starting)))
    if waiting or starting:
        begun = min(starts, default=view.created_at.timestamp() if view.created_at else None)
        timer = _clock(now - begun) if seen.ran and begun is not None else ""
        head = _stage(_spinning(now), "provision", detail, timer, width)
        return (head, *_waiting(view, seen, waiting, now, width))
    ends = [times.bootstrap for times in _times(seen, members) if times.bootstrap is not None]
    took = max(ends) - min(starts) if ends else None
    return (_stage(Text("✓ ", style="green"), "provision", detail, _span(took) if took is not None else "", width, active=False),)


def _bootstrap(view: ComputeView, seen: Seen, now: float, width: int) -> tuple[Text, ...]:
    live = _live(view)
    setting = tuple(node for node in live if node.state in _BOOTSTRAP)
    up = tuple(node for node in live if node.state in _UP)
    if not setting and not (seen.opening and seen.setup and up):
        return ()
    detail = Text(f"{len(up)}/{len(live) or view.initial} ready")
    if not setting:
        return (_stage(_IDLE, "bootstrap", detail, "", width, active=False),)
    starts = [times.bootstrap for times in _times(seen, _members(view, seen)) if times.bootstrap is not None]
    timer = _clock(now - min(starts)) if starts else ""
    return (_stage(_spinning(now), "bootstrap", detail, timer, width), *_steps(view, seen, setting, now, width))


def _run(view: ComputeView, seen: Seen, now: float, width: int) -> tuple[Text, ...]:
    live = _live(view)
    if not (seen.ran or (live and not seen.setup and all(node.state in _UP for node in live))):
        return ()
    counts = view.task_counts or _counted(view.tasks)
    busy = counts.running + counts.queued > 0
    head = _fit(Text.assemble(_spinning(now) if busy else _IDLE, _tally(counts) or Text("idle")), width)
    gauges = _gauges(tuple(_latest(node) for node in view.nodes if node.state in _UP))
    return (head,) if gauges is None else (head, _fit(Text.assemble("  ", gauges), width))


def _waiting(view: ComputeView, seen: Seen, waiting: Sequence[NodeView], now: float, width: int) -> tuple[Text, ...]:
    """The machines still on their way that say what they are doing: one line each, or one per thing they are doing."""
    talking = sorted((node for node in waiting if node.progress), key=lambda node: node.rank)
    if len(talking) <= LISTED:
        return tuple(_fit(Text.assemble(_cell(view, node.rank), "  ", node.progress or "", *_progress(node.completion)), width) for node in talking)
    groups = groupby(sorted(talking, key=lambda node: node.progress or ""), key=lambda node: node.progress or "")
    return tuple(
        _grouped(view, step, nodes, _slowest(nodes, seen, now, provisioning=True), lambda node: _percent(node.completion), width)
        for step, grouped in groups
        for nodes in (tuple(grouped),)
    )


def _steps(view: ComputeView, seen: Seen, setting: Sequence[NodeView], now: float, width: int) -> tuple[Text, ...]:
    """The nodes in bootstrap: one line each with the step and its last line of output, or one line per step past :data:`LISTED`."""
    if len(setting) <= LISTED:
        listed = sorted(setting, key=lambda node: node.rank)
        column = max(len(_step(node)) for node in listed) + 2
        return tuple(
            _fit(Text.assemble(_cell(view, node.rank), "  ", f"{_step(node):<{column}}", _dim(_said(node)) or ""), width)
            for node in listed
        )
    ordered = sorted(setting, key=lambda node: (_STEPS.get(_step(node), len(_STEPS)), _step(node)))
    return tuple(
        _grouped(view, step, nodes, _slowest(nodes, seen, now, provisioning=False), _said, width)
        for step, grouped in groupby(ordered, key=_step)
        for nodes in (tuple(grouped),)
    )


def _grouped(view: ComputeView, step: str, nodes: Sequence[NodeView], slowest: NodeView, detail: Callable[[NodeView], str], width: int) -> Text:
    """One line for every node on a step: how many, the one that has been at it longest, and what it last said."""
    who = Text.assemble("node ", _ranked(slowest.rank)) if len(nodes) == 1 else Text.assemble(f"{len(nodes)} nodes", _DOT, "slowest ", _ranked(slowest.rank))
    return _fit(Text.assemble("    ", f"{step:<16}", _dotted(who, _dim(detail(slowest) or None))), width)


def _slowest(nodes: Sequence[NodeView], seen: Seen, now: float, *, provisioning: bool) -> NodeView:
    def since(node: NodeView) -> float:
        times = seen.times.get(node.id, Times())
        begun = times.provision if provisioning else times.bootstrap
        return now if begun is None else begun

    return min(nodes, key=lambda node: (since(node), node.rank))


def _note(members: Sequence[NodeView], seen: Seen, *, under_way: bool) -> Text:
    """Which nodes this setup is for, once the compute has been up: the ones replacing a rank that went away, and the ones added to it."""

    def told(reason: Reason, doing: str, done: str) -> str | None:
        ranks = sorted(node.rank for node in members if seen.times.get(node.id, Times()).reason == reason)
        if not ranks:
            return None
        nodes = f"node {ranks[0]}" if len(ranks) == 1 else f"{len(ranks)} nodes"
        return f"{doing} {nodes}" if under_way else f"{nodes} {done}"

    phrases = (told("replacing", "replacing", "replaced"), told("adding", "adding", "added"))
    return Text.assemble(*(part for phrase in phrases if phrase for part in (_DOT, (phrase, "bright_black"))))


def _title(view: ComputeView, seen: Seen) -> Text:
    where = " ".join(part for part in (view.provider, view.region) if part)
    return _dotted(Text("skyward", style="bold"), Text(view.name or view.id, style="bold"), Text(_described(view, seen.fleet)), Text(where))


def _described(view: ComputeView, fleet: Fleet) -> str:
    """The fleet as one phrase: how many nodes, of what, bought how — ``4× A100 80GB spot``."""
    if view.accelerator:
        label = _accelerator_label(view.accelerator)
        memory = f" {fleet.memory_gb:.0f}GB" if fleet.memory_gb and "GB" not in label else ""
        what = f"{f'{view.accelerator_count}×' if view.accelerator_count > 1 else ''}{label}{memory}"
    elif view.cpus:
        what = f"{view.cpus} vCPU" + (f" {view.memory_gb:g} GB" if view.memory_gb else "")
    else:
        what = ""
    nodes = f"{fleet.nodes}× {what}" if what else f"{fleet.nodes} node" + ("" if fleet.nodes == 1 else "s")
    return f"{nodes} {fleet.market}".rstrip()


def _fleet(view: ComputeView, previous: Fleet) -> Fleet:
    """The fleet as the view has it, or as it last had it: the machines of a compute being deleted are no longer there to ask."""
    live = _live(view)
    if not live:
        return previous if previous.nodes else Fleet(nodes=view.initial, market=_market((), view.allocation))
    totals = [node.metrics["gpu_mem_total_mb"][-1] for node in live if node.metrics.get("gpu_mem_total_mb")]
    memory = totals[0] / max(view.accelerator_count, 1) / 1024 if totals else previous.memory_gb
    return Fleet(nodes=len(live), market=_market(live, view.allocation), memory_gb=memory)


def _market(live: Sequence[NodeView], allocation: str) -> str:
    match sorted({node.market for node in live if node.market}):
        case ["spot"]:
            return "spot"
        case ["on_demand"]:
            return "on-demand"
        case ["on_demand", "spot"]:
            return "spot + on-demand"
        case _:
            return {"spot": "spot", "on_demand": "on-demand"}.get(allocation, "")


def _tally(counts: TaskCounts) -> Text | None:
    phrases = (
        (counts.running, "running", ""),
        (counts.queued, "queued", ""),
        (counts.succeeded, "done", ""),
        (counts.failed, "failed", "red"),
        (counts.timed_out, "timed out", "red"),
    )
    shown = [Text(f"{count:,} {word}", style=style) for count, word, style in phrases if count]
    return _dotted(*shown) if shown else None


def _counted(tasks: Sequence[TaskView]) -> TaskCounts:
    """The counts from the tasks the view holds, for a view the API has not been read into yet."""
    states = Counter(task.state for task in tasks)
    return TaskCounts(
        queued=states["queued"],
        running=states["running"],
        succeeded=states["succeeded"],
        failed=states["failed"],
        cancelled=states["cancelled"],
        timed_out=states["timed_out"],
        indeterminate=states["indeterminate"],
    )


def _gauges(readings: Sequence[Mapping[str, float]]) -> Text | None:
    """What the nodes read, one reading per node: the mean, and the lowest and highest node beside it when there is more than one."""

    def values(read: Callable[[Mapping[str, float]], float | None]) -> tuple[float, ...]:
        return tuple(value for reading in readings if (value := read(reading)) is not None)

    def memory(reading: Mapping[str, float]) -> float | None:
        if "mem" in reading:
            return reading["mem"]
        total = reading.get("mem_total_mb")
        return reading["mem_used_mb"] / total * 100 if total and "mem_used_mb" in reading else None

    def gauge(name: str, read: tuple[float, ...], unit: str) -> Text | None:
        if not read:
            return None
        low, high = f"{min(read):.0f}", f"{max(read):.0f}"
        return Text.assemble((f"{name} ", "bright_black"), f"{fmean(read):.0f}{unit}", (f" {low}–{high}", "bright_black") if low != high else "")

    used = values(lambda reading: reading["gpu_mem_mb"] / 1024 if "gpu_mem_mb" in reading else None)
    totals = values(lambda reading: reading["gpu_mem_total_mb"] / 1024 if "gpu_mem_total_mb" in reading else None)
    shown = [
        part
        for part in (
            gauge("gpu", values(lambda reading: reading.get("gpu_util")), "%"),
            gauge("vram", used, f"/{fmean(totals):.0f} GB" if totals else " GB"),
            gauge("cpu", values(lambda reading: reading.get("cpu")), "%"),
            gauge("mem", values(memory), "%"),
        )
        if part is not None
    ]
    return Text("   ").join(shown) if shown else None


def _latest(node: NodeView) -> Mapping[str, float]:
    return MappingProxyType({name: history[-1] for name, history in node.metrics.items() if history})


def _spans(nodes: frozenset[str], times: Mapping[str, Times]) -> Spans:
    """How long the nodes took to provision and to bootstrap: from the first of them to begin a stage to the last of them to leave it."""
    seen = [times[node] for node in nodes if node in times]

    def took(begins: Callable[[Times], float | None], ends: Callable[[Times], float | None]) -> float | None:
        started = [value for each in seen if (value := begins(each)) is not None]
        finished = [value for each in seen if (value := ends(each)) is not None]
        return max(finished) - min(started) if started and finished else None

    return Spans(
        provision=took(lambda each: each.provision, lambda each: each.bootstrap),
        bootstrap=took(lambda each: each.bootstrap, lambda each: each.ready),
    )


def _took(spans: Spans | None) -> tuple[Text, ...]:
    if spans is None:
        return ()
    stages = (("provision", spans.provision), ("bootstrap", spans.bootstrap))
    return tuple(Text.assemble((f"{stage} ", "bright_black"), _span(value)) for stage, value in stages if value is not None)


def _stage(icon: Text, name: str, detail: Text, timer: str, width: int, *, active: bool = True) -> Text:
    return _justified(Text.assemble(icon, Text(f"{name:<{_LABEL}}", style="bold" if active else ""), detail), Text(timer), width)


def _justified(left: Text, right: Text, width: int) -> Text:
    """``left`` cut to fit, and ``right`` against the right edge."""
    room = width - right.cell_len - (2 if right.plain else 0)
    left.truncate(max(room, 1), overflow="ellipsis")
    return Text.assemble(left, " " * (width - left.cell_len - right.cell_len), right) if right.plain else left


def _fit(line: Text, width: int) -> Text:
    line.truncate(width, overflow="ellipsis")
    return line


def _from(view: ComputeView, row: NodeView | None, body: Text) -> Text:
    """A scrollback line, behind the rank of the node it is about when there is one to name."""
    if row is None:
        return body
    width = len(str(max(node.rank for node in view.nodes)))
    return Text.assemble(_ranked(row.rank, width), (" │ ", "bright_black"), body)


def _cell(view: ComputeView, rank: int) -> Text:
    width = len(str(max((node.rank for node in view.nodes), default=0)))
    return Text.assemble("    ", _ranked(rank, width))


def _ranked(rank: int, width: int = 0) -> Text:
    return Text(f"{rank:>{width}}", style=_HUES[rank % len(_HUES)])


def _dotted(*parts: Text | None) -> Text:
    return Text.assemble(*(piece for index, part in enumerate(part for part in parts if part) for piece in ((_DOT, part) if index else (part,))))


def _dim(text: str | None) -> Text | None:
    return Text(text, style="bright_black") if text else None


def _dollars(amount: float | None) -> str | None:
    return f"${amount:.2f}" if amount is not None else None


def _progress(completion: float | None) -> tuple[Text | str, ...]:
    return () if completion is None else ("  ", _bar(completion), f" {completion * 100:.0f}%")


def _percent(completion: float | None) -> str:
    return "" if completion is None else f"{completion * 100:.0f}%"


def _step(node: NodeView) -> str:
    if node.state == "connecting":
        return _PHASE_LABELS["connecting"]
    latest = next((phase.name for phase in reversed(node.phases) if phase.started), None)
    return _PHASE_LABELS.get(latest, latest) if latest else "waiting"


def _said(node: NodeView) -> str:
    return node.tail[-1] if node.tail else ""


def _function(task: TaskView | None) -> str:
    return task.function if task is not None and task.function else "task"


def _live(view: ComputeView) -> tuple[NodeView, ...]:
    return tuple(node for node in view.nodes if node.state in _LIVE)


def _members(view: ComputeView, seen: Seen) -> tuple[NodeView, ...]:
    return tuple(node for node in view.nodes if node.id in seen.setup)


def _times(seen: Seen, nodes: Sequence[NodeView]) -> tuple[Times, ...]:
    return tuple(seen.times[node.id] for node in nodes if node.id in seen.times)


def _row(view: ComputeView, node: str) -> NodeView | None:
    return next((row for row in view.nodes if row.id == node), None)


def _task(view: ComputeView, task: str) -> TaskView | None:
    return next((row for row in view.tasks if row.id == task), None)


def _spinning(now: float) -> Text:
    return Text(f"{_SPINNER_FRAMES[int(now * 10) % len(_SPINNER_FRAMES)]} ", style="blue")


_IDLE = Text("· ", style="bright_black")


def _clock(seconds: float) -> str:
    """A timer still running: ``1:58``, or ``2h14m`` past the hour."""
    whole = max(0, int(seconds))
    hours, rest = divmod(whole, 3600)
    return f"{hours}h{rest // 60:02d}m" if hours else f"{rest // 60}:{rest % 60:02d}"


def _span(seconds: float) -> str:
    """A time that is over: ``38s``, ``1m28s``, ``2h14m``."""
    whole = max(0, round(seconds))
    hours, rest = divmod(whole, 3600)
    minutes, rest = divmod(rest, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{rest:02d}s" if minutes else f"{rest}s"


class _LogSink(logging.Handler):
    def __init__(self, console: Console) -> None:
        super().__init__(logging.WARNING)
        self._console = console
        self.setFormatter(logging.Formatter("%(module)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._console.print(Text(self.format(record), style="yellow"))
        except Exception:
            self.handleError(record)


@contextmanager
def _quiet(console: Console) -> Iterator[None]:
    """Route warnings into the dashboard for the length of the block.

    Two loggers, because the daemon's records never reach the root one: they go to
    skyward's own non-propagating logger, and a record with nowhere to land would be
    printed to stderr by ``logging``'s last resort, through the live display.
    """
    sink = _LogSink(console)
    saved = [(target, target.handlers[:], target.level) for target in (logging.getLogger(), logging.getLogger(LOGGER_NAME))]
    for target, _, _ in saved:
        target.handlers = [sink]
        target.setLevel(logging.WARNING)
    try:
        yield
    finally:
        for target, handlers, level in saved:
            target.handlers, target.level = handlers, level
