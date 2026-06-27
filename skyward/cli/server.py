"""sky server — manage the Skyward HTTP server.

``start`` runs as a detached daemon by default: it spawns ``uvicorn`` as
a separate process group, polls ``/health`` until ready, prints the URL
and PID, and returns. The PID is persisted in ``~/.skyward/server.pid``
and stdout/stderr go to ``~/.skyward/server.log``. Use ``--foreground``
to keep the server attached to the terminal (the old behavior, useful
with ``--reload`` for dev).

``stop`` POSTs ``/shutdown`` and cleans up the PID file. ``status`` hits
``/health``.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated

import httpx
from cyclopts import Parameter

from . import server_app
from ._client import format_http_error, make_client, resolve_server_url
from ._output import console, print_status, print_table

_RUNTIME_DIR = Path.home() / ".skyward"
_PID_FILE = _RUNTIME_DIR / "server.pid"
_LOG_FILE = _RUNTIME_DIR / "server.log"

_HEALTH_POLL_INTERVAL = 0.2
_HEALTH_TIMEOUT_DEFAULT = 30.0
_STOP_POLL_INTERVAL = 0.1
_STOP_TIMEOUT_DEFAULT = 10.0


# ── PID file helpers ─────────────────────────────────────────────


def _read_pid() -> int | None:
    """Return the PID stored in :data:`_PID_FILE`, or ``None`` if absent or unreadable."""
    if not _PID_FILE.exists():
        return None
    try:
        return int(_PID_FILE.read_text().strip())
    except (ValueError, OSError):
        return None


def _is_alive(pid: int) -> bool:
    """Return whether the given PID is alive in the OS process table."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists; we just can't signal it
    return True


def _write_pid(pid: int) -> None:
    _RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    _PID_FILE.write_text(str(pid))


def _clear_pid() -> None:
    _PID_FILE.unlink(missing_ok=True)


# ── Health polling ───────────────────────────────────────────────


def _wait_for_health(url: str, *, timeout: float) -> bool:
    """Poll ``GET {url}/health`` until 200, or return ``False`` on timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{url}/health", timeout=1.0)
            if r.status_code == 200:
                return True
        except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException):
            pass
        time.sleep(_HEALTH_POLL_INTERVAL)
    return False


def _wait_for_exit(pid: int, *, timeout: float) -> bool:
    """Wait for ``pid`` to disappear, or return ``False`` on timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _is_alive(pid):
            return True
        time.sleep(_STOP_POLL_INTERVAL)
    return False


# ── Foreground vs daemon ─────────────────────────────────────────


def _run_foreground(host: str, port: int, *, reload: bool) -> None:
    import uvicorn

    config = uvicorn.Config("skyward.server:create_app", factory=True, host=host, port=port, reload=reload)
    uvicorn.Server(config).run()


def _spawn_daemon(host: str, port: int) -> subprocess.Popen:
    """Spawn a detached uvicorn process; stdout/stderr → ``_LOG_FILE``."""
    _RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    log = _LOG_FILE.open("ab")  # noqa: SIM115 — handed to subprocess; closed on its exit
    cmd = [
        sys.executable, "-m", "uvicorn",
        "skyward.server:create_app", "--factory",
        "--host", host,
        "--port", str(port),
    ]
    return subprocess.Popen(
        cmd,
        stdout=log,
        stderr=log,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )


# ── Commands ─────────────────────────────────────────────────────


@server_app.command(name="start")
def start_server(
    *,
    host: Annotated[str, Parameter(name="--host", help="Bind host")] = "127.0.0.1",
    port: Annotated[int, Parameter(name="--port", help="Bind port")] = 7590,
    reload: Annotated[
        bool,
        Parameter(name="--reload", help="Enable auto-reload (implies --foreground)"),
    ] = False,
    foreground: Annotated[
        bool,
        Parameter(name="--foreground", help="Stay attached to the terminal (Ctrl+C to stop)"),
    ] = False,
    timeout: Annotated[
        float,
        Parameter(name="--timeout", help="Seconds to wait for /health when daemonizing"),
    ] = _HEALTH_TIMEOUT_DEFAULT,
) -> None:
    """Start the Skyward HTTP server.

    Defaults to daemon mode: spawns ``uvicorn`` as a detached process,
    waits for ``/health``, prints the PID and log path, and returns.
    Use ``--foreground`` to keep the server attached.
    """
    if foreground or reload:
        _run_foreground(host, port, reload=reload)
        return

    existing = _read_pid()
    if existing is not None and _is_alive(existing):
        console.print(
            f"[yellow]Server already running (pid {existing}).[/yellow] "
            "Use [bold]sky server stop[/bold] first."
        )
        raise SystemExit(1)
    if existing is not None:
        _clear_pid()  # stale

    proc = _spawn_daemon(host, port)
    _write_pid(proc.pid)

    target = f"http://{host}:{port}"
    if not _wait_for_health(target, timeout=timeout):
        console.print(
            f"[red]Server did not become healthy within {timeout:.0f}s.[/red] "
            f"See [bold]{_LOG_FILE}[/bold]"
        )
        with contextlib.suppress(ProcessLookupError):
            os.kill(proc.pid, signal.SIGTERM)
        _clear_pid()
        raise SystemExit(1)

    console.print(f"[green]Server running at {target}[/green] [dim](pid {proc.pid})[/dim]")
    console.print(f"[dim]Logs:[/dim]  {_LOG_FILE}")
    console.print("[dim]Stop:[/dim]  sky server stop")


@server_app.command(name="stop")
def stop_server(
    *,
    url: Annotated[str | None, Parameter(name="--url", help="Server URL")] = None,
    timeout: Annotated[
        float,
        Parameter(name="--timeout", help="Seconds to wait for the daemon to exit"),
    ] = _STOP_TIMEOUT_DEFAULT,
) -> None:
    """Trigger a graceful shutdown on the server."""
    target = resolve_server_url(url)
    try:
        with make_client(url) as client:
            r = client.post("/shutdown")
    except httpx.ConnectError:
        pid = _read_pid()
        if pid is not None and not _is_alive(pid):
            _clear_pid()
            console.print("[yellow]No server running (cleaned stale pid file).[/yellow]")
            return
        console.print(f"[red]Could not reach server at {target}[/red]")
        raise SystemExit(1) from None
    if r.status_code != 202:
        console.print(f"[red]{format_http_error(r)}[/red]")
        raise SystemExit(1)

    console.print(f"[green]Shutdown requested at {target}[/green]")

    pid = _read_pid()
    if pid is not None and not _wait_for_exit(pid, timeout=timeout):
        console.print(
            f"[yellow]Daemon still alive after {timeout:.0f}s; pid file kept.[/yellow]"
        )
        return
    _clear_pid()


@server_app.command(name="status")
def server_status(
    *,
    url: Annotated[str | None, Parameter(name="--url", help="Server URL")] = None,
    json: Annotated[bool, Parameter(name="--json", help="JSON output")] = False,
) -> None:
    """Show server health, version, and live counts."""
    target = resolve_server_url(url)
    try:
        with make_client(url) as client:
            r = client.get("/health")
    except httpx.ConnectError:
        if json:
            import json as _json
            import sys as _sys

            _sys.stdout.write(_json.dumps({"url": target, "status": "unreachable"}) + "\n")
            return
        print_status("Server", "fail", f"unreachable at {target}")
        raise SystemExit(1) from None

    if r.status_code != 200:
        console.print(f"[red]{format_http_error(r)}[/red]")
        raise SystemExit(1)

    payload = r.json()
    if json:
        import json as _json
        import sys as _sys

        _sys.stdout.write(_json.dumps({"url": target, **payload}) + "\n")
        return

    print_table(
        ["URL", "Status", "Version", "Pools", "Executions"],
        [(target, payload["status"], payload["version"], payload["pools"], payload["executions"])],
    )
