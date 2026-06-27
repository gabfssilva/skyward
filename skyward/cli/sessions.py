"""sky — Colab-style session verbs over the HTTP server's ``/compute`` API.

A "session" is a named compute pool. These top-level verbs (``new``,
``sessions``, ``status``, ``stop``) wrap the same routes as ``sky compute
*`` but track a client-side *current session* so ``-s/--session`` can be
omitted when unambiguous.
"""

from __future__ import annotations

import sys
from typing import Annotated

import httpx
from cyclopts import Parameter

from . import app
from ._client import format_http_error, make_client, resolve_server_url
from ._output import console, print_status, print_table
from ._session_store import (
    clear_current_session,
    read_current_session,
    resolve_session,
    write_current_session,
)
from .compute import _POOL_COLUMNS, _create_session, _pool_row, _present_pool_created


@app.command(name="new")
def new_session(
    name: Annotated[str | None, Parameter(help="Session name")] = None,
    *,
    pool: Annotated[str | None, Parameter(name="--pool", help="Named pool from skyward.toml")] = None,
    provider: Annotated[str | None, Parameter(name="--provider", help="Provider type (aws, vastai, runpod, …)")] = None,
    region: Annotated[str | None, Parameter(name="--region", help="Override region on the resolved spec")] = None,
    nodes: Annotated[int | None, Parameter(name="--nodes", help="Override node count")] = None,
    accelerator: Annotated[str | None, Parameter(name="--accelerator", help="Override accelerator (e.g. A100, H100)")] = None,
    allocation: Annotated[str | None, Parameter(name="--allocation", help="Override allocation")] = None,
    pip: Annotated[list[str] | None, Parameter(name="--pip", help="Extra pip packages (repeatable)")] = None,
    apt: Annotated[list[str] | None, Parameter(name="--apt", help="Extra apt packages (repeatable)")] = None,
    python: Annotated[str | None, Parameter(name="--python", help="Python version on the worker")] = None,
    watch: Annotated[bool, Parameter(name="--watch", help="Follow pool events live until ready/failed")] = False,
    url: Annotated[str | None, Parameter(name="--url", help="Server URL")] = None,
    json: Annotated[bool, Parameter(name="--json", help="JSON output")] = False,
) -> None:
    """Create a named session (an N-node compute pool) and make it current."""
    info = _create_session(
        pool=pool, name=name, provider=provider, region=region, nodes=nodes,
        accelerator=accelerator, allocation=allocation, pip=pip, apt=apt,
        python=python, url=url,
    )
    write_current_session(info["name"])
    _present_pool_created(info, target=resolve_server_url(url), watch=watch, json=json)


@app.command(name="sessions")
def list_sessions(
    *,
    url: Annotated[str | None, Parameter(name="--url", help="Server URL")] = None,
    json: Annotated[bool, Parameter(name="--json", help="JSON output")] = False,
) -> None:
    """List all sessions on the server (marks the current one)."""
    target = resolve_server_url(url)
    try:
        with make_client(url) as client:
            r = client.get("/compute")
    except httpx.ConnectError:
        console.print(f"[red]Could not reach server at {target}[/red]")
        raise SystemExit(1) from None
    if r.status_code != 200:
        console.print(f"[red]{format_http_error(r)}[/red]")
        raise SystemExit(1)

    pools = r.json()
    current = read_current_session()
    if json:
        import json as _json

        sys.stdout.write(
            _json.dumps([{**p, "current": p["name"] == current} for p in pools]) + "\n",
        )
        return
    if not pools:
        console.print("[dim]No sessions[/dim]")
        return
    print_table(
        ["", *_POOL_COLUMNS],
        [("→" if p["name"] == current else "", *_pool_row(p)) for p in pools],
    )


@app.command(name="status")
def session_status(
    session: Annotated[str | None, Parameter(name=("-s", "--session"), help="Session name")] = None,
    *,
    url: Annotated[str | None, Parameter(name="--url", help="Server URL")] = None,
    json: Annotated[bool, Parameter(name="--json", help="JSON output")] = False,
) -> None:
    """Show one session's status and its per-node table."""
    name = resolve_session(session, url)
    target = resolve_server_url(url)
    try:
        with make_client(url) as client:
            r = client.get(f"/compute/{name}")
            nodes_resp = client.get(f"/compute/{name}/nodes")
    except httpx.ConnectError:
        console.print(f"[red]Could not reach server at {target}[/red]")
        raise SystemExit(1) from None

    if r.status_code == 404:
        console.print(f"[red]Session {name!r} not found[/red]")
        raise SystemExit(1)
    if r.status_code != 200:
        console.print(f"[red]{format_http_error(r)}[/red]")
        raise SystemExit(1)

    info = r.json()
    nodes = nodes_resp.json()["nodes"] if nodes_resp.status_code == 200 else []
    if json:
        import json as _json

        sys.stdout.write(_json.dumps({**info, "nodes": nodes}) + "\n")
        return

    print_table(_POOL_COLUMNS, [_pool_row(info)])
    if nodes:
        print_table(
            ["Rank", "Head", "Status", "Instance", "Address"],
            [
                (
                    n["rank"],
                    "✓" if n["is_head"] else "",
                    n["status"],
                    n["instance_id"] or "-",
                    f"{n['ssh_user']}@{n['ip']}:{n['ssh_port']}" if n["ip"] else "-",
                )
                for n in nodes
            ],
        )


@app.command(name="stop")
def stop_session(
    session: Annotated[str | None, Parameter(name=("-s", "--session"), help="Session name")] = None,
    *,
    url: Annotated[str | None, Parameter(name="--url", help="Server URL")] = None,
) -> None:
    """Tear down a session (real cloud teardown) and clear it if current."""
    name = resolve_session(session, url)
    target = resolve_server_url(url)
    try:
        with make_client(url) as client:
            r = client.delete(f"/compute/{name}")
    except httpx.ConnectError:
        console.print(f"[red]Could not reach server at {target}[/red]")
        raise SystemExit(1) from None

    if read_current_session() == name:
        clear_current_session()

    if r.status_code == 404:
        console.print(f"[red]Session {name!r} not found[/red]")
        raise SystemExit(1)
    if r.status_code != 204:
        console.print(f"[red]{format_http_error(r)}[/red]")
        raise SystemExit(1)
    print_status(f"Session '{name}'", "ok", "stopped")
