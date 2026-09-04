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
import time
from collections.abc import Mapping
from contextlib import suppress
from typing import Literal, Protocol, TextIO

from skyward.core.client import Client
from skyward.core.view import ComputeView, EventCallback, decoded, observe, refresh, refresh_tasks
from skyward.shared import lifecycle
from skyward.shared.events import (
    ComputeDegraded,
    ComputeDeletionFailed,
    ConsoleEvent,
    Event,
    NodeEvent,
    ProgressEvent,
    TaskEvent,
    progressed,
)
from skyward.shared.schemas import (
    Compute as ComputeResource,
)
from skyward.shared.schemas import Function, Node, Page, Task

type ConsoleMode = Literal["rich", "log"]

POLL = 2.0
"""How stale the API half of the view may get before an event prompts a re-read."""


class Watcher(Protocol):
    """A console attached to the pool for its whole life.

    The :class:`Observer` opens it once, feeds it every event with the view
    folded up to it, tells it when the API half of the view was re-read, and
    closes it when the stream ends. A user's callback is narrower — one
    callable, events only — and does not need this shape.
    """

    def opened(self, view: ComputeView) -> None: ...

    def event(self, event: Event, view: ComputeView) -> None: ...

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
        self._names: dict[str, str] = {}
        self._fetched = 0.0

    async def follow(self) -> None:
        with suppress(Exception):
            self._view = await self._fetch(self._view)
        await asyncio.to_thread(self._open)
        try:
            async for _, payload in self._client.events(self._compute):
                if (event := decoded(payload)) is None:
                    continue
                view = observe(self._view, event)
                if self._stale(event):
                    with suppress(Exception):
                        view = await self._fetch(view)
                self._view = view
                await asyncio.to_thread(self._dispatch, event, view)
        except Exception as exc:
            print(f"skyward: the event stream stopped ({exc})", file=sys.stderr, flush=True)
        finally:
            await asyncio.to_thread(self._close)

    def _stale(self, event: Event) -> bool:
        """Whether this event obsoletes the API half of the view.

        A node or task transition always does — an address, a price, a timing
        just appeared somewhere only a read can see. Anything else does only
        once the last read has aged past ``POLL``, which turns the steady drip
        of gauges and cost into the poll the panel used to run for itself.
        """
        match event:
            case NodeEvent() | TaskEvent():
                return True
            case _:
                return time.monotonic() - self._fetched > POLL

    async def _fetch(self, view: ComputeView) -> ComputeView:
        compute = await self._client.call("GET", f"/v1/computes/{self._compute}", ComputeResource)
        nodes = await self._client.call("GET", f"/v1/computes/{self._compute}/nodes", Page[Node])
        tasks: Page[Task] = Page(items=())
        with suppress(Exception):
            tasks = await self._client.call("GET", "/v1/tasks", Page[Task], compute=self._compute, limit=200)
        self._fetched = time.monotonic()
        return refresh_tasks(refresh(view, compute, nodes), tasks, await self._names_for(tasks))

    async def _names_for(self, tasks: Page[Task]) -> Mapping[str, str]:
        """Each function's name, asked once; a function that cannot be named is asked once too."""
        for sha in {task.function for task in tasks.items} - self._names.keys():
            try:
                self._names[sha] = (await self._client.call("GET", f"/v1/functions/{sha}", Function)).name or ""
            except Exception:
                self._names[sha] = ""
        return self._names

    def _open(self) -> None:
        for one in self._watchers:
            one.opened(self._view)

    def _dispatch(self, event: Event, view: ComputeView) -> None:
        for one in self._watchers:
            one.event(event, view)
        for callback in self._callbacks:
            try:
                callback(event, view)
            except Exception as exc:
                print(f"skyward: a callback raised ({exc})", file=sys.stderr, flush=True)

    def _close(self) -> None:
        for one in self._watchers:
            with suppress(Exception):
                one.closed(self._view)

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
