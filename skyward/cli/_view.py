"""Live renderer for ``sky compute view`` and ``sky compute create --watch``.

Consumes the SSE stream at ``GET /compute/{name}/events`` and renders the
same Rich layout the in-process console actor uses (``_LiveFooter`` from
:mod:`skyward.actors.console.view`). Events are reconstructed via
:mod:`skyward.server.wire` and fed to a local :class:`SessionProjection`,
which gives the renderer the same ``PoolView`` it would see in-process.

Modes
-----
- ``json_mode=True``: emit NDJSON (snapshot + each event on its own line).
- ``once=True``: render the snapshot once and exit.
- default: Rich Live until ``done`` or Ctrl+C.

Returns an exit code: ``0`` on ``ready``, ``1`` on ``failed``,
``2`` on stream/connection errors.
"""

from __future__ import annotations

import json as _json
import sys
from dataclasses import replace
from types import MappingProxyType
from typing import TYPE_CHECKING

import httpx

from skyward.actors.console.state import _State
from skyward.actors.console.view import (
    WARNING_STYLE,
    _emit,
    _emit_task,
    _LiveFooter,
    _node_label,
    _print_provisioning_error,
    _render_summary,
    _ssh_url,
    _state_from_pool_view,
)
from skyward.api.events import Error, Log, Node, Pool, Task
from skyward.api.projection import SessionProjection
from skyward.server.wire import event_from_json, pool_view_from_json

from ._output import console as _cli_console

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from rich.console import Console


type Frame = tuple[str, dict | None]
"""One parsed SSE frame: ``(event_type, json_payload)``."""


# ── SSE parsing ──────────────────────────────────────────────────


async def iter_sse(url: str, name: str) -> AsyncIterator[Frame]:
    """Async-iterate frames from ``GET /compute/{name}/events``.

    Parameters
    ----------
    url : str
        Base URL of the Skyward HTTP server.
    name : str
        Pool name.

    Yields
    ------
    Frame
        ``(event_type, payload)`` tuples; ``payload`` is the JSON-decoded
        body, or ``None`` for empty data lines.

    Raises
    ------
    RuntimeError
        If the server returns a non-200 status.
    """
    async with (
        httpx.AsyncClient(base_url=url, timeout=httpx.Timeout(30.0, read=None)) as client,
        client.stream("GET", f"/compute/{name}/events") as response,
    ):
        if response.status_code != 200:
            body = await response.aread()
            raise RuntimeError(f"events stream returned {response.status_code}: {body[:200]!r}")
        event_type = ""
        data_lines: list[str] = []
        async for raw in response.aiter_lines():
            line = raw.rstrip("\r")
            if line == "":
                if event_type:
                    payload = _json.loads("".join(data_lines)) if data_lines else None
                    yield event_type, payload
                event_type = ""
                data_lines = []
                continue
            if line.startswith(":"):
                continue  # comment frame (keepalive)
            if line.startswith("event:"):
                event_type = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].lstrip())


# ── Mode 1: NDJSON ───────────────────────────────────────────────


async def render_ndjson(url: str, name: str) -> int:
    """Stream the SSE feed verbatim as NDJSON to stdout."""
    try:
        async for event_type, payload in iter_sse(url, name):
            sys.stdout.write(_json.dumps({"event": event_type, "data": payload}) + "\n")
            sys.stdout.flush()
            if event_type == "done":
                status = (payload or {}).get("status")
                return 1 if status == "failed" else 0
    except (httpx.ConnectError, RuntimeError) as exc:
        _cli_console.print(f"[red]{exc}[/red]")
        return 2
    return 0


# ── Mode 2: snapshot only ────────────────────────────────────────


async def render_once(url: str, name: str) -> int:
    """Render the initial snapshot through ``_LiveFooter`` once and exit."""
    from rich.console import Console

    console = Console(stderr=True)
    try:
        async for event_type, payload in iter_sse(url, name):
            if event_type == "snapshot":
                if payload is None:
                    _cli_console.print(
                        "[dim]No snapshot available — pool exists but has not started yet.[/dim]"
                    )
                    return 0
                view = pool_view_from_json(payload)
                footer = _LiveFooter()
                footer.state = _state_from_pool_view(view)
                console.print(footer)
                return 0
            if event_type == "done":
                _cli_console.print(f"[red]Pool {name!r} not available: {payload}[/red]")
                return 1
    except (httpx.ConnectError, RuntimeError) as exc:
        _cli_console.print(f"[red]{exc}[/red]")
        return 2
    return 0


# ── Mode 3: Rich Live (parity with console actor) ────────────────


def _dispatch_event(console: Console, event: object, state: _State) -> None:
    """Print a one-line update for a domain event.

    Mirrors ``skyward.actors.console.actor._print_event`` but kept here so
    the CLI stays decoupled from the actor module. All emitters come from
    ``skyward.actors.console.view`` (``_emit``, ``_emit_task``, etc.).
    """
    match event:
        case Pool.ProvisionFailed(reason=reason):
            _print_provisioning_error(console, reason)
        case Node.Ready(node_id=nid):
            label = _node_label(state, nid)
            _emit(console, label, "✓ Joined", "green bold", link=_ssh_url(state, nid))
        case Node.Lost(node_id=nid, reason=reason):
            _emit(console, "error", f"Node {nid} lost: {reason}", "red")
        case Node.ConnectionFailed(error=error):
            _emit(console, "error", f"SSH failed: {error}", "red")
        case Node.Preempted(reason=reason):
            _emit(console, "error", f"Preempted: {reason}", "red")
        case Node.WorkerFailed(error=error):
            _emit(console, "error", f"Worker failed: {error}", "red")
        case Node.Bootstrap.Failed(node_id=nid, phase=phase, error=err):
            label = _node_label(state, nid)
            _emit(console, label, f"✗ {phase}: {err}", "red", link=_ssh_url(state, nid))
        case Task.Queued(name=tname, kind="broadcast"):
            n = len(state.instances)
            _emit_task(console, "skyward", "queued", f"{tname} → all {n} nodes")
        case Task.Queued(name=tname):
            _emit_task(console, "skyward", "queued", tname)
        case Task.Completed(node_id=nid, elapsed=elapsed):
            label = _node_label(state, nid)
            _emit_task(console, label, "done", f"in {elapsed:.1f}s", link=_ssh_url(state, nid))
        case Task.Failed(node_id=nid):
            label = _node_label(state, nid)
            _emit_task(console, label, "failed", "", link=_ssh_url(state, nid))
        case Error.Occurred(message=message, fatal=fatal):
            style = "red bold" if fatal else "red"
            _emit(console, "error", message, style)
        case Pool.Stopped():
            pass


def _state_from_projection(
    projection: SessionProjection,
    name: str,
    progress: dict[int, str],
) -> _State:
    """Build the renderer's ``_State`` from the local projection."""
    pool = projection.view.pools.get(name)
    state = _state_from_pool_view(pool) if pool else _State(total_nodes=0)
    if progress:
        state = replace(state, progress_lines=MappingProxyType(dict(progress)))
    return state


def _handle_log(
    console: Console,
    log: Log.Emitted,
    progress: dict[int, str],
    state: _State,
) -> None:
    """Apply a ``Log.Emitted`` event: progress overwrite or one-shot emit."""
    nid = log.node_id
    if log.overwrite:
        progress[nid] = log.message
        return
    if nid in progress:
        _emit(console, _node_label(state, nid), progress.pop(nid))
    _emit(console, _node_label(state, nid), log.message)


def _flush_progress(console: Console, progress: dict[int, str], state: _State) -> None:
    """Emit any pending progress lines and clear them."""
    for nid, content in list(progress.items()):
        _emit(console, _node_label(state, nid), content)
    progress.clear()


async def render_live(url: str, name: str) -> int:
    """Render the SSE feed live with the same layout as the console actor."""
    from rich.console import Console
    from rich.live import Live

    console = Console(stderr=True)
    projection = SessionProjection()
    footer = _LiveFooter()
    progress: dict[int, str] = {}
    exit_code = 0

    try:
        with Live(
            footer,
            console=console,
            refresh_per_second=8,
            screen=False,
            redirect_stdout=False,
            redirect_stderr=False,
        ) as live:
            try:
                async for event_type, payload in iter_sse(url, name):
                    if event_type == "snapshot":
                        if payload is not None:
                            projection.seed(name, pool_view_from_json(payload))
                            footer.state = _state_from_projection(projection, name, progress)
                        continue

                    if event_type == "done":
                        status = (payload or {}).get("status")
                        state = _state_from_projection(projection, name, progress)
                        live.stop()
                        if status == "failed":
                            exit_code = 1
                            err = (payload or {}).get("error") or "(no error message)"
                            _print_provisioning_error(console, err)
                        elif status == "deleted":
                            _emit(console, "skyward", "Pool deleted", "bright_black")
                        elif status == "stopping":
                            _emit(console, "skyward", "Stopping...", WARNING_STYLE)
                        console.print(_render_summary(state))
                        return exit_code

                    event = event_from_json(payload or {})
                    if event is None:
                        continue

                    if isinstance(event, Log.Emitted):
                        state = _state_from_projection(projection, name, progress)
                        _handle_log(console, event, progress, state)
                        footer.state = _state_from_projection(projection, name, progress)
                        continue

                    projection.handle(event)
                    state = _state_from_projection(projection, name, progress)

                    if isinstance(event, (Pool.Stopped, Pool.ProvisionFailed)):
                        _flush_progress(console, progress, state)
                    elif isinstance(event, Node.Lost) and event.node_id in progress:
                        _emit(console, _node_label(state, event.node_id), progress.pop(event.node_id))

                    _dispatch_event(console, event, state)
                    footer.state = _state_from_projection(projection, name, progress)
            except KeyboardInterrupt:
                live.stop()
                _emit(console, "skyward", "Disconnected (Ctrl+C). Pool keeps running.", "bright_black")
                return 0
    except (httpx.ConnectError, RuntimeError) as exc:
        _cli_console.print(f"[red]{exc}[/red]")
        return 2
    return exit_code


async def render(url: str, name: str, *, json_mode: bool, once: bool) -> int:
    """Dispatch to the right renderer based on flags."""
    if json_mode:
        return await render_ndjson(url, name)
    if once:
        return await render_once(url, name)
    return await render_live(url, name)
