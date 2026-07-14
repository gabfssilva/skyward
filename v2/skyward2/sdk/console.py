"""What the pool says while it is working.

The events are already in the store and already have a stream; this is the half
that reads them. It renders what a person watching a terminal wants to know —
where the machines got to, and what the code on them printed — and drops the rest,
which is not lost, it is in the log.

Everything goes to stderr. A script's stdout is its own, and a pool that wrote
progress into it would corrupt every pipeline it was ever put in.
"""

from __future__ import annotations

import asyncio
import sys
from typing import TextIO

import msgspec

from skyward2.sdk.client import Client


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
        try:
            async for event, payload in self._client.events(self._compute):
                if line := render(event, msgspec.json.decode(payload, type=dict[str, str])):
                    print(line, file=self.out, flush=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"skyward: the event stream stopped ({exc})", file=self.out, flush=True)


def render(event: str, payload: dict[str, str]) -> str | None:
    """One event, as a line, or nothing if it is not worth a line.

    A task's outcome gets no line: the caller is holding the result or the exception
    and has a better account of it than a log line could give. What the machines are
    doing is a different matter — nobody else is going to say it.
    """
    node = payload.get("node", "")
    match event:
        case "node.console":
            return f"{node} │ {payload.get('content', '')}"
        case "node.failed" | "compute.degraded":
            return f"{node or payload.get('compute', '')} │ {event}: {payload.get('error') or 'failed'}"
        case "compute.provisioning" | "compute.ready" | "compute.deleted":
            return f"{payload.get('compute', '')} │ {event.removeprefix('compute.')}"
        case _ if event.startswith("node."):
            return f"{node} │ {event.removeprefix('node.')}"
        case _:
            return None
