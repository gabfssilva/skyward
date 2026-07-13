"""sky monitor — full-screen Textual dashboard for a live server session."""

from __future__ import annotations

import contextlib
from typing import Annotated

import httpx
from cyclopts import Parameter

from . import app
from ._client import format_http_error, make_client, resolve_server_url
from ._output import console


@app.command(name="monitor")
def monitor(
    name: Annotated[str, Parameter(help="Session/pool name on the server")],
    *,
    dark: Annotated[bool, Parameter(name="--dark", help="Start in the dark theme")] = False,
    url: Annotated[str | None, Parameter(name="--url", help="Server URL")] = None,
) -> None:
    """Open the full-screen TUI monitor for a running session.

    Attaches to ``GET /compute/{name}/events`` (SSE), feeds a local
    :class:`~skyward.api.projection.SessionProjection`, and renders the live
    cluster dashboard.  Navigate with the arrow keys, ``enter`` to open a
    node, ``t`` to toggle theme; quit with ``q`` — the pool keeps running.
    """
    import asyncio

    target = resolve_server_url(url)

    # Pre-flight so an unknown pool or unreachable server fails with a clean
    # message instead of an empty dashboard once the screen is taken over.
    try:
        with make_client(url) as client:
            r = client.get(f"/compute/{name}")
    except httpx.ConnectError:
        console.print(f"[red]Could not reach server at {target}[/red]")
        raise SystemExit(1) from None
    if r.status_code == 404:
        console.print(f"[red]Pool {name!r} not found[/red]")
        raise SystemExit(1)
    if r.status_code != 200:
        console.print(f"[red]{format_http_error(r)}[/red]")
        raise SystemExit(1)

    asyncio.run(_run(target, name, dark=dark))


async def _run(target: str, name: str, *, dark: bool) -> None:
    """Drive the SSE pump and the Textual app on one event loop.

    Both the event pump and the app share this loop, so the projection
    mutations (pump) and the dashboard reads (app timers) are serialized —
    no locking is needed for this topology.
    """
    import asyncio

    from skyward.api.projection import SessionProjection
    from skyward.server.wire import event_from_json, pool_view_from_json
    from skyward.tui.app import SkywardTUI
    from skyward.tui.sources import ProjectionSource

    from ._view import iter_sse

    projection = SessionProjection()
    source = ProjectionSource(projection, pool_name=name)
    tui = SkywardTUI(source, start_dark=dark, tick_interval=1.0)

    async def pump() -> None:
        with contextlib.suppress(httpx.ConnectError, RuntimeError):
            async for event_type, payload in iter_sse(target, name):
                if event_type == "snapshot":
                    if payload is not None:
                        projection.seed(name, pool_view_from_json(payload))
                elif event_type == "done":
                    return
                elif (event := event_from_json(payload or {})) is not None:
                    projection.handle(event)

    pump_task = asyncio.create_task(pump())
    try:
        await tui.run_async()
    finally:
        pump_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pump_task
        source.close()
