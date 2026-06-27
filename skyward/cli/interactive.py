"""sky console / sky repl — interactive PTY into a session node.

Both fetch the node's SSH coordinates from the server, then open a direct
PTY to the node (see :mod:`skyward.cli._interactive`). Phase 1 is
co-located + key-only: the CLI must run on the same host as the server and
the node's private key must exist locally; password-only nodes error out.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import httpx
from cyclopts import Parameter

from . import app
from ._client import format_http_error, make_client, resolve_server_url
from ._interactive import open_pty
from ._output import console

_REPL_COMMAND = "cd /opt/skyward && exec /opt/skyward/.venv/bin/python"


def _open_node_pty(name: str, rank: int, url: str | None, command: str | None) -> None:
    target = resolve_server_url(url)
    try:
        with make_client(url) as client:
            r = client.get(f"/compute/{name}/nodes")
    except httpx.ConnectError:
        console.print(f"[red]Could not reach server at {target}[/red]")
        raise SystemExit(1) from None

    if r.status_code == 404:
        console.print(f"[red]Session {name!r} not found[/red]")
        raise SystemExit(1)
    if r.status_code == 409:
        try:
            status = r.json().get("status", "?")
        except ValueError:
            status = "?"
        console.print(f"[red]session not ready ({status})[/red]")
        raise SystemExit(1)
    if r.status_code != 200:
        console.print(f"[red]{format_http_error(r)}[/red]")
        raise SystemExit(1)

    nodes = r.json()["nodes"]
    node = next((n for n in nodes if n["rank"] == rank), None)
    if node is None:
        console.print(f"[red]No node with rank {rank}[/red]")
        raise SystemExit(1)

    key_path = node.get("ssh_key_path")
    if not key_path or not Path(key_path).exists():
        if node.get("has_password"):
            console.print("[red]password-only nodes are not supported in interactive mode (Phase 1)[/red]")
        else:
            console.print(
                "[red]interactive mode requires the CLI to run on the same host "
                "as the server (Phase 1 limitation)[/red]"
            )
        raise SystemExit(1)

    ip = node.get("ip")
    if not ip:
        console.print(f"[red]node rank {rank} has no address yet[/red]")
        raise SystemExit(1)

    code = asyncio.run(
        open_pty(ip, node["ssh_port"], node["ssh_user"], key_path, None, command),
    )
    raise SystemExit(code)


@app.command(name="console")
def console_(
    name: Annotated[str, Parameter(help="Session/pool name on the server")],
    *,
    rank: Annotated[int, Parameter(name="--rank", help="Node rank (default head)")] = 0,
    url: Annotated[str | None, Parameter(name="--url", help="Server URL")] = None,
) -> None:
    """Open a raw interactive shell (PTY) on a node of a running session."""
    _open_node_pty(name, rank, url, None)


@app.command(name="repl")
def repl(
    name: Annotated[str, Parameter(help="Session/pool name on the server")],
    *,
    rank: Annotated[int, Parameter(name="--rank", help="Node rank (default head)")] = 0,
    url: Annotated[str | None, Parameter(name="--url", help="Server URL")] = None,
) -> None:
    """Open an interactive Python REPL inside a node's worker virtualenv."""
    _open_node_pty(name, rank, url, _REPL_COMMAND)
