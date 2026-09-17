"""What the pool says while it is working.

The events are already in the store and already have a stream; this is the half
that reads them. It renders what a person watching a terminal wants to know —
where the machines got to, and what the code on them printed — and drops the rest,
which is not lost, it is in the log.

Everything goes to stderr. A script's stdout is its own, and a pool that wrote
progress into it would corrupt every pipeline it was ever put in.

Colour is applied only when the stream is a terminal: the same lines piped to a
file or another process arrive plain, so a redirect never inherits escape codes.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from contextlib import suppress
from dataclasses import replace
from functools import partial, reduce
from typing import Literal, Protocol, TextIO

from skyward.api.v1 import ComputeResource, Page, TaskResource
from skyward.core.client import Client
from skyward.core.view import TASKS, ComputeView, EventCallback, decoded, observe, refresh, refresh_tasks
from skyward.shared import lifecycle
from skyward.shared.events import (
    ComputeDegraded,
    ComputeDeletionFailed,
    ConsoleEvent,
    Event,
    GenerationCreated,
    NodeEvent,
    ProgressEvent,
    TaskEvent,
    progressed,
)

type ConsoleMode = Literal["rich", "log"]

POLL = 2.0
"""The closest two reads of the API half of the view come to each other.

An event that moved what only a read can show asks for one, and a burst of them
gets one: whatever asks while a read is out, or sooner than this after the last,
is answered by the next.
"""

QUIET = 10.0
"""The longest the API half of the view goes unread, whether anything asked or not.

Not all of it moves with an event. A machine is bought and given an address, and
a task is queued, cancelled or timed out, without the stream saying so — and a
task queued behind one that runs for an hour would otherwise go unshown for the
hour.
"""


class Watcher(Protocol):
    """A console attached to the pool for its whole life.

    The :class:`Observer` opens it once, feeds it every event with the view
    folded up to it, hands it the view again whenever the API half is re-read
    between events, and closes it when the stream ends. A user's callback is
    narrower — one callable, events only — and does not need this shape.
    """

    def opened(self, view: ComputeView) -> None: ...

    def event(self, event: Event, view: ComputeView) -> None: ...

    def refreshed(self, view: ComputeView) -> None: ...

    def closed(self, view: ComputeView) -> None: ...


def watcher(out: TextIO | None = None, *, mode: ConsoleMode = "rich") -> Watcher:
    """Select the Rich live view or the line log: the live view only on a terminal, and only when Rich is installed."""
    if mode == "log" or not (out or sys.stderr).isatty():
        return Console(out)
    try:
        from skyward.core.live import RichConsole
    except ImportError:
        return Console(out)
    return RichConsole(out)


class Observer:
    """One SSE stream, one fold, everybody watching.

    The consoles and the user's callbacks all hang off this one consumer: the
    stream is read once, folded once into a :class:`ComputeView`, and each
    subscriber is handed the same value. A callback that raises is reported and
    skipped — a broken observer must not take the training run with it.

    The API half of the view is read beside the stream, never in its way. A
    stream that stopped for a read on every node and task event fell behind a
    busy compute until the daemon hung up on it, and a replayed history asked for
    a read per event it had ever recorded. What an event carries is folded from
    the event; only what it cannot carry — a node's address, a task's timings, a
    compute's new bounds — asks for a read, and every ask made before the read
    goes out is the same read.
    """

    def __init__(
        self,
        client: Client,
        compute: str,
        watchers: tuple[Watcher, ...] = (),
        callbacks: tuple[EventCallback, ...] = (),
    ) -> None:
        self._client = client
        self._compute = compute
        self._watchers = watchers
        self._callbacks = callbacks
        self._view = ComputeView(id=compute)
        self._asked = asyncio.Event()
        self._overtaken: list[Event] | None = None
        """What the stream moved while a read was out, or ``None`` while no read is."""
        self._turn = asyncio.Lock()

    async def follow(self) -> None:
        with suppress(Exception):
            await self._read()
        await self._tell(self._open)
        try:
            async with asyncio.TaskGroup() as group:
                reading = group.create_task(self._reread())
                try:
                    async for _, payload in self._client.events(self._compute):
                        if (event := decoded(payload)) is not None:
                            await self._fold(event)
                finally:
                    reading.cancel()
        except* Exception as stopped:
            print(f"skyward: the event stream stopped ({stopped.exceptions[0]})", file=sys.stderr, flush=True)
        finally:
            await self._tell(self._close)

    async def _fold(self, event: Event) -> None:
        self._view = observe(self._view, event)
        if _asks(event):
            self._asked.set()
        if self._overtaken is not None and _moves(event):
            self._overtaken.append(event)
        await self._tell(partial(self._dispatch, event))

    async def _reread(self) -> None:
        """Read the API half again once something asks: no sooner than ``POLL`` after the last read, and no later than ``QUIET``."""
        while True:
            await asyncio.sleep(POLL)
            with suppress(TimeoutError):
                async with asyncio.timeout(QUIET - POLL):
                    await self._asked.wait()
            self._asked.clear()
            try:
                await self._read()
            except Exception:
                continue
            await self._tell(self._refreshed)

    async def _read(self) -> None:
        """Read the API half, and lay it over the view as the view is once the read is back.

        The stream goes on while the read is out, so a read can land older than
        the view it lands on: a node the stream saw become ready, read while it was
        still bootstrapping. What the stream moved in the meantime is folded again
        over the read, so a read never takes back what an event already said — all
        but the errors, which the stream noted once and must not note twice.
        """
        overtaken: list[Event] = []
        self._overtaken = overtaken
        try:
            compute = await self._client.call("GET", f"/v1/computes/{self._compute}", ComputeResource)
            tasks: Page[TaskResource] = Page(items=(), next_cursor=None, total=None)
            with suppress(Exception):
                tasks = await self._client.call("GET", "/v1/tasks", Page[TaskResource], compute=self._compute, limit=TASKS)
        finally:
            self._overtaken = None
        read = refresh_tasks(refresh(self._view, compute), tasks)
        self._view = replace(reduce(observe, overtaken, read), errors=read.errors)

    async def _tell(self, call: Callable[[ComputeView], None]) -> None:
        """Hand the view as it is now to the watchers, off the loop, one hand-off at a time.

        Watchers draw in a thread so that a slow terminal never holds the loop up,
        and both the stream and the reads hand them views: taking turns is what
        keeps a view from being drawn over by an older one.
        """
        async with self._turn:
            await asyncio.to_thread(call, self._view)

    def _open(self, view: ComputeView) -> None:
        for one in self._watchers:
            one.opened(view)

    def _dispatch(self, event: Event, view: ComputeView) -> None:
        for one in self._watchers:
            one.event(event, view)
        for callback in self._callbacks:
            try:
                callback(event, view)
            except Exception as exc:
                print(f"skyward: a callback raised ({exc})", file=sys.stderr, flush=True)

    def _refreshed(self, view: ComputeView) -> None:
        for one in self._watchers:
            one.refreshed(view)

    def _close(self, view: ComputeView) -> None:
        for one in self._watchers:
            with suppress(Exception):
                one.closed(view)

RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"

_STATE = {
    "requested": "\033[90m",       # grey — nobody has bought it yet
    "provisioning": "\033[33m",    # yellow
    "connecting": "\033[36m",      # cyan
    "bootstrapping": "\033[34m",   # blue
    "ready": "\033[32m",           # green
    "draining": "\033[33m",        # yellow
    "lost": "\033[31m",            # red
    "deleting": "\033[90m",        # grey
    "deleted": "\033[90m",         # grey
    "failed": "\033[31m",          # red
    "degraded": "\033[31m",        # red
}
"""How each lifecycle word is coloured — by what it means, not alphabetically."""

_NODE_HUES = ("\033[36m", "\033[35m", "\033[33m", "\033[34m", "\033[32m", "\033[95m", "\033[94m")
"""One stable colour per node id, so a broadcast's lines stay visually sorted."""


class Console:
    """The compute's log, on the terminal, one line per event worth one."""

    def __init__(self, out: TextIO | None = None) -> None:
        self._out = out

    @property
    def out(self) -> TextIO:
        """Resolved per line, never captured.

        ``sys.stderr`` is not a constant — a notebook rebinds it, a test harness
        replaces it between phases — and a console holding the one it was born with
        writes its lines into a stream nobody is reading any more.
        """
        return self._out or sys.stderr

    def opened(self, view: ComputeView) -> None:
        return None

    def event(self, event: Event, view: ComputeView) -> None:
        if line := render(event, self.out.isatty()):
            print(line, file=self.out, flush=True)

    def refreshed(self, view: ComputeView) -> None:
        return None

    def closed(self, view: ComputeView) -> None:
        return None


def render(event: Event, color: bool = False) -> str | None:
    """One event, as a line, or nothing if it is not worth a line.

    A task's outcome gets no line: the caller is holding the result or the exception
    and has a better account of it than a log line could give. What the machines are
    doing is a different matter — nobody else is going to say it. The gauges get
    none either: a reading every couple of seconds is a graph, not a log.

    A machine still short of an address is the exception to that, and the reason is
    that there is nothing else on the line: a container host pulling an image sits in
    ``provisioning`` for minutes, and its progress is only sent when it moves, so what
    would be a graph anywhere else is here the only sign the pool is not hung.
    """
    match event:
        case ConsoleEvent(node=node, content=content):
            return f"{_who(node, color)} {_sep(color)} {content}"
        case ProgressEvent(node=node, progress=progress, completion=completion):
            return f"{_who(node, color)} {_sep(color)} {_dim(progressed(progress, completion), color)}"
        case NodeEvent(node=node, state="failed" | "lost" as state, error=error) if error:
            return f"{_who(node, color)} {_sep(color)} {_badge(state, color)} {_dim(error, color)}"
        case NodeEvent(node=node, state=state):
            return f"{_who(node, color)} {_sep(color)} {_badge(state, color)}"
        case ComputeDegraded(compute=compute, error=error):
            return f"{_who(compute, color)} {_sep(color)} {_badge('degraded', color)} {_dim(error, color)}"
        case ComputeDeletionFailed(compute=compute, error=error):
            return f"{_who(compute, color)} {_sep(color)} {_badge('deleting', color)} {_dim(error, color)}"
        case _ if (state := lifecycle.leads(event)):
            return f"{_who(event.compute, color)} {_sep(color)} {_badge(state, color)}"
        case _:
            return None


def _asks(event: Event) -> bool:
    """Whether the event moved what only a read can show: a node's address, a task's timings, a compute's bounds."""
    match event:
        case NodeEvent() | TaskEvent() | GenerationCreated():
            return True
        case _:
            return False


def _moves(event: Event) -> bool:
    """Whether folding the event writes what a read writes too — a node's, a task's or the compute's state.

    Those are the events a read that left before them would take back, so they
    are the ones folded again over it.
    """
    match event:
        case NodeEvent() | TaskEvent():
            return True
        case _:
            return lifecycle.leads(event) is not None


def _who(ident: str, color: bool) -> str:
    """The id, tinted with its own stable colour so nodes never blur together."""
    if not color or not ident:
        return ident
    hue = _NODE_HUES[hash(ident) % len(_NODE_HUES)]
    return f"{hue}{ident}{RESET}"


def _badge(state: str, color: bool) -> str:
    """A lifecycle word, coloured by what the machine is doing."""
    if not color:
        return state
    return f"{_STATE.get(state, '')}{BOLD}{state}{RESET}"


def _sep(color: bool) -> str:
    return f"{DIM}│{RESET}" if color else "│"


def _dim(text: str, color: bool) -> str:
    return f"{DIM}{text}{RESET}" if color else text
