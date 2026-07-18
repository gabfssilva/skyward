"""sky log — a compute's recorded execution history.

The daemon keeps one log per compute and hands it out over SSE, replayed from
the beginning and then followed. That is the whole source: node output, phase
marks and task outcomes are events there, so printing the history and following
it live are the same call under a different stopping rule.

The stream never ends on its own, so a command that is not following stops once
the replay goes quiet — ``--idle`` is how long quiet has to last.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Callable
from contextlib import aclosing
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from skyward.sdk.client import Client

from . import log_app
from ._client import call
from ._output import Output, cell, dump, render

type Event = tuple[str, dict[str, str]]
type Echo = Callable[[Event], None]

IDLE = 1.0


def _who(payload: dict[str, str]) -> str | None:
    return payload.get("node") or payload.get("task") or payload.get("compute")


def _detail(name: str, payload: dict[str, str]) -> str:
    match name:
        case "node.console":
            return payload.get("content", "")
        case "node.phase":
            return " ".join(filter(None, (payload.get("event"), payload.get("phase"), payload.get("error"))))
        case _:
            return payload.get("error", "")


def _echo(output: Output) -> Echo:
    def line(event: Event) -> None:
        name, payload = event
        match output:
            case "json":
                sys.stdout.write(json.dumps({"event": name, **payload}) + "\n")
            case "table":
                sys.stdout.write("  ".join(cell(value) for value in (name, _who(payload), _detail(name, payload))) + "\n")
        sys.stdout.flush()

    return line


async def _read(client: Client, compute: str, idle: float | None, echo: Echo | None) -> list[Event]:
    """Replay the log, and keep following it when ``idle`` is None."""
    events: list[Event] = []
    async with aclosing(client.events(compute)) as stream:
        feed = stream.__aiter__()
        while True:
            try:
                async with asyncio.timeout(idle):
                    name, payload = await anext(feed)
            except (TimeoutError, StopAsyncIteration):
                return events
            event: Event = (name, json.loads(payload))
            events.append(event)
            if echo:
                echo(event)


def _markdown(events: list[Event]) -> str:
    """Console output grouped by node, with everything else listed after it."""
    order: list[str] = []
    console: dict[str, list[str]] = {}
    rest: list[Event] = []

    for name, payload in events:
        if name != "node.console":
            rest.append((name, payload))
            continue
        node = payload.get("node", "-")
        if node not in console:
            order.append(node)
            console[node] = []
        console[node].append(payload.get("content", ""))

    out = ["# Execution log", ""]
    for node in order:
        out += [f"## node `{node}`", "", "```", *console[node], "```", ""]
    if rest:
        out += ["## Events", "", *(f"- **{name}** {json.dumps(payload)}" for name, payload in rest)]
    return "\n".join(out) + "\n"


def _serialize(events: list[Event], suffix: str) -> str:
    match suffix:
        case ".md" | ".markdown":
            return _markdown(events)
        case ".jsonl" | ".json" | "":
            return "".join(json.dumps({"event": name, **payload}) + "\n" for name, payload in events)
        case _:
            raise SystemExit(f"unknown export format '{suffix}' — use .jsonl or .md")


@log_app.default
def show_log(
    compute: Annotated[str, Parameter(help="Compute id")],
    *,
    follow: Annotated[bool, Parameter(name=("-f", "--follow"), help="Keep streaming after the replay")] = False,
    limit: Annotated[int | None, Parameter(name=("-n", "--limit"), help="Last N events only")] = None,
    idle: Annotated[float, Parameter(name="--idle", help="Seconds of quiet that end a replay")] = IDLE,
    url: Annotated[str | None, Parameter(name="--url", help="Daemon URL")] = None,
    database: Annotated[Path | None, Parameter(name="--database", help="Embedded daemon database")] = None,
    output: Annotated[Output, Parameter(name="--output", help="table or json")] = "table",
) -> None:
    """Print a compute's event log, replayed from the start."""
    echo = _echo(output) if follow else None
    events = call(lambda client: _read(client, compute, None if follow else idle, echo), url=url, database=database)

    if follow:
        return

    tail = events[-limit:] if limit else events
    match output:
        case "json":
            dump([{"event": name, **payload} for name, payload in tail])
        case "table":
            render(["event", "who", "detail"], [(name, _who(payload), _detail(name, payload)) for name, payload in tail])


@log_app.command(name="export")
def export_log(
    compute: Annotated[str, Parameter(help="Compute id")],
    file: Annotated[Path, Parameter(help="Destination file — .jsonl or .md")],
    *,
    limit: Annotated[int | None, Parameter(name=("-n", "--limit"), help="Last N events only")] = None,
    idle: Annotated[float, Parameter(name="--idle", help="Seconds of quiet that end a replay")] = IDLE,
    url: Annotated[str | None, Parameter(name="--url", help="Daemon URL")] = None,
    database: Annotated[Path | None, Parameter(name="--database", help="Embedded daemon database")] = None,
) -> None:
    """Write a compute's event log to a file, as JSONL or Markdown."""
    events = call(lambda client: _read(client, compute, idle, None), url=url, database=database)
    tail = events[-limit:] if limit else events

    file.write_text(_serialize(tail, file.suffix.lower()))
    sys.stdout.write(f"{file}  {len(tail)} events\n")


__all__ = ["export_log", "show_log"]
