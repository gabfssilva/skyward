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
from typing import Protocol, TextIO

import msgspec

from skyward.sdk.client import Client

GAUGES = frozenset({"node.metrics", "compute.cost"})
"""The readings, which the log has no line for and could not read anyway.

Every other event's payload is strings; a gauge's is the number it carries. The
log renders neither, and it has to say so before it decodes rather than after —
a payload it cannot read is not a reason to stop reading the stream.
"""


class Follower(Protocol):
    """Whatever is watching the pool: the line log, or the live panel above it."""

    async def follow(self) -> None: ...


def watcher(client: Client, compute: str, out: TextIO | None = None) -> Follower:
    """The richest view the terminal can hold.

    The panel needs a terminal to pin itself to and a keyboard's terminal to read
    from, and it needs the extra installed. Missing any of the three, the pool
    still says everything — one line at a time, which is what a pipe wanted anyway.
    """
    stream = out or sys.stderr
    if stream.isatty() and sys.stdin.isatty():
        try:
            from skyward.sdk.live import Dashboard
        except ImportError:
            return Console(client, compute, out)
        return Dashboard(client, compute, out)
    return Console(client, compute, out)

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
    """The compute's log, on the terminal, for as long as the pool is open."""

    def __init__(self, client: Client, compute: str, out: TextIO | None = None) -> None:
        self._client = client
        self._compute = compute
        self._out = out

    @property
    def out(self) -> TextIO:
        """Resolved per line, never captured.

        ``sys.stderr`` is not a constant — a notebook rebinds it, a test harness
        replaces it between phases — and a console holding the one it was born with
        writes its lines into a stream nobody is reading any more.
        """
        return self._out or sys.stderr

    async def follow(self) -> None:
        color = self.out.isatty()
        try:
            async for event, payload in self._client.events(self._compute):
                if event in GAUGES:
                    continue
                if line := render(event, msgspec.json.decode(payload, type=dict[str, str]), color):
                    await asyncio.to_thread(print, line, file=self.out, flush=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"skyward: the event stream stopped ({exc})", file=self.out, flush=True)


def render(event: str, payload: dict[str, str], color: bool = False) -> str | None:
    """One event, as a line, or nothing if it is not worth a line.

    A task's outcome gets no line: the caller is holding the result or the exception
    and has a better account of it than a log line could give. What the machines are
    doing is a different matter — nobody else is going to say it.
    """
    node = payload.get("node", "")
    match event:
        case "node.console":
            return f"{_who(node, color)} {_sep(color)} {payload.get('content', '')}"
        case "node.failed" | "compute.degraded":
            who = node or payload.get("compute", "")
            detail = payload.get("error") or "failed"
            return f"{_who(who, color)} {_sep(color)} {_badge(event.split('.')[-1], color)} {_dim(detail, color)}"
        case "compute.provisioning" | "compute.ready" | "compute.deleted":
            who = payload.get("compute", "")
            return f"{_who(who, color)} {_sep(color)} {_badge(event.removeprefix('compute.'), color)}"
        case "node.metrics" | "node.phase":
            return None
        case _ if event.startswith("node."):
            return f"{_who(node, color)} {_sep(color)} {_badge(event.removeprefix('node.'), color)}"
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
