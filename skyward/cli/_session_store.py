"""Client-side current-session pointer for the Skyward CLI.

Mirrors the PID-file lifecycle in :mod:`skyward.cli.server`: a small file
under ``~/.skyward`` records the session the user last created or
switched to, so commands can default ``-s/--session`` to it. "Session" is
CLI vocabulary for a server-side compute pool — there are no ``/sessions``
routes; everything resolves against ``/compute``.
"""

from __future__ import annotations

from pathlib import Path

import httpx

from ._client import format_http_error, make_client, resolve_server_url
from ._output import console

SESSION_FILE = Path.home() / ".skyward" / "current-session"


def read_current_session() -> str | None:
    """Return the persisted current-session name, or ``None`` if unset."""
    if not SESSION_FILE.exists():
        return None
    try:
        name = SESSION_FILE.read_text().strip()
    except OSError:
        return None
    return name or None


def write_current_session(name: str) -> None:
    """Persist *name* as the current session."""
    SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    SESSION_FILE.write_text(name)


def clear_current_session() -> None:
    """Remove the current-session pointer if present."""
    SESSION_FILE.unlink(missing_ok=True)


def live_sessions(url: str | None) -> list[str]:
    """Return the names of sessions registered on the server.

    Exits the process on an unreachable server or a non-200 response,
    matching the established CLI error pattern.
    """
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
    return [p["name"] for p in r.json()]


def resolve_session(name: str | None, url: str | None) -> str:
    """Resolve the target session name.

    Precedence: explicit *name* > persisted current session if still live
    on the server > the single live session when exactly one exists. Exits
    with guidance when zero or multiple sessions exist and none was given.
    """
    if name:
        return name

    live = live_sessions(url)
    current = read_current_session()
    if current is not None:
        if current in live:
            return current
        clear_current_session()

    if not live:
        console.print("[red]No sessions. Create one with [bold]sky new[/bold].[/red]")
        raise SystemExit(1)
    if len(live) == 1:
        write_current_session(live[0])
        return live[0]
    console.print(
        "[red]Multiple sessions; pass [bold]-s <name>[/bold].[/red] "
        f"Live: {', '.join(sorted(live))}"
    )
    raise SystemExit(1)
