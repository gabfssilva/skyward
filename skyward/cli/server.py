"""``sky server`` — run the daemon, or find out whether one is running.

The daemon is the Litestar app at :mod:`skyward.server.http.app`; this only starts a
process around it. ``start`` detaches by default and records the pid, because a
control plane that dies with the terminal that launched it is not a control
plane. ``--foreground`` keeps it attached, which is what a dev loop wants.

``stop`` signals the pid rather than asking the daemon to end itself: there is
no shutdown endpoint, and a control plane should not offer one — anything that
can reach the API could then take the whole plane down.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated

import httpx
from cyclopts import Parameter

from skyward.cli import server_app
from skyward.cli._client import HOST, PORT, call, resolve
from skyward.cli._output import Output, render
from skyward.core.client import Client
from skyward.shared.schemas import Liveness

RUNTIME_DIR = Path.home() / ".skyward"
PID_FILE = RUNTIME_DIR / "server.pid"
LOG_FILE = RUNTIME_DIR / "server.log"

TARGET = "skyward.server.http.app:daemon"
POLL_SECONDS = 0.2


def endpoint(url: str | None, host: str, port: int) -> str:
    """Return the URL to probe: an explicit one, else the address given to probe.

    ``resolve`` already falls back to where ``start`` binds, but ``status`` is the
    one command that can be pointed at a *different* bind, so the flags win over
    that default and only an explicit URL — flag or environment — wins over them.
    """
    if url or os.environ.get("SKYWARD_URL"):
        return resolve(url)
    return f"http://{host}:{port}"


def pid() -> int | None:
    """Return the recorded pid, or None when there is no readable pidfile."""
    try:
        return int(PID_FILE.read_text().strip())
    except (OSError, ValueError):
        return None


def alive(process: int) -> bool:
    """Return whether the process exists, signalling nothing to find out."""
    try:
        os.kill(process, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def probe(client: Client) -> bool:
    """Return whether ``/v1/health/live`` answers affirmatively."""
    try:
        return (await client.call("GET", "/v1/health/live", Liveness)).live
    except (httpx.TransportError, OSError):
        return False


def live(target: str) -> bool:
    """Return whether a daemon answers at ``target``."""
    return call(probe, url=target)


def _wait_live(target: str, timeout: float) -> bool:
    async def watch(client: Client) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            if await probe(client):
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(POLL_SECONDS)

    return call(watch, url=target)


def _wait_exit(process: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while alive(process):
        if time.monotonic() >= deadline:
            return False
        time.sleep(POLL_SECONDS)
    return True


def _require_uvicorn() -> None:
    if importlib.util.find_spec("uvicorn") is None:
        raise SystemExit("the daemon needs an ASGI server: pip install uvicorn")


def _environment(database: Path | None) -> dict[str, str]:
    if database is None:
        return dict(os.environ)
    return {**os.environ, "SKYWARD_DATABASE": str(database)}


def _foreground(host: str, port: int, database: Path | None) -> None:
    import uvicorn

    if database is not None:
        os.environ["SKYWARD_DATABASE"] = str(database)
    uvicorn.run(TARGET, host=host, port=port, factory=True)


def _spawn(host: str, port: int, database: Path | None) -> int:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_FILE.open("ab")  # noqa: SIM115
    command = [sys.executable, "-m", "uvicorn", TARGET, "--factory", "--host", host, "--port", str(port)]
    process = subprocess.Popen(
        command, stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True, env=_environment(database)
    )
    PID_FILE.write_text(str(process.pid))
    return process.pid


@server_app.command(name="start")
def start(
    *,
    host: Annotated[str, Parameter(help="Bind address")] = HOST,
    port: Annotated[int, Parameter(help="Bind port")] = PORT,
    foreground: Annotated[bool, Parameter(help="Stay attached to the terminal")] = False,
    timeout: Annotated[float, Parameter(help="Seconds to wait for the daemon to answer")] = 30.0,
    database: Annotated[Path | None, Parameter(help="SQLite path (default: ~/.skyward/skyward.sqlite)")] = None,
) -> None:
    """Start the Skyward daemon.

    Parameters
    ----------
    host
        Address to bind.
    port
        Port to bind.
    foreground
        Run attached, ending with the terminal, instead of detaching.
    timeout
        How long to wait for liveness before giving up on a detached start.
    database
        The SQLite file the daemon keeps its state in.
    """
    _require_uvicorn()

    if foreground:
        _foreground(host, port, database)
        return

    if (running := pid()) and alive(running):
        raise SystemExit(f"already running (pid {running}) — sky server stop")

    PID_FILE.unlink(missing_ok=True)
    process = _spawn(host, port, database)

    if not _wait_live(f"http://{host}:{port}", timeout):
        if alive(process):
            os.kill(process, signal.SIGTERM)
        PID_FILE.unlink(missing_ok=True)
        raise SystemExit(f"no answer within {timeout:.0f}s — see {LOG_FILE}")

    print(f"http://{host}:{port} (pid {process})")
    print(f"logs: {LOG_FILE}")


@server_app.command(name="stop")
def stop(
    *,
    timeout: Annotated[float, Parameter(help="Seconds to wait for the process to exit")] = 10.0,
) -> None:
    """Stop the daemon this machine started.

    Parameters
    ----------
    timeout
        How long to wait for the process to leave before reporting it stayed.
    """
    match pid():
        case None:
            raise SystemExit("no pidfile — nothing to stop")
        case int(process) if not alive(process):
            PID_FILE.unlink(missing_ok=True)
            print(f"not running (cleared stale pid {process})")
        case int(process):
            os.kill(process, signal.SIGTERM)
            if not _wait_exit(process, timeout):
                raise SystemExit(f"pid {process} still alive after {timeout:.0f}s")
            PID_FILE.unlink(missing_ok=True)
            print(f"stopped (pid {process})")


@server_app.command(name="status")
def status(
    *,
    url: Annotated[str | None, Parameter(help="Daemon URL")] = None,
    host: Annotated[str, Parameter(help="Bind address to probe when no URL resolves")] = HOST,
    port: Annotated[int, Parameter(help="Bind port to probe when no URL resolves")] = PORT,
    output: Annotated[Output, Parameter(name=["--output", "-o"], help="Rendering")] = "table",
) -> None:
    """Report the recorded pid and whether a daemon answers.

    Parameters
    ----------
    url
        Overrides ``SKYWARD_URL``.
    host
        Address to probe when neither ``--url`` nor the environment says.
    port
        Port to probe when neither ``--url`` nor the environment says.
    output
        ``table`` for a person, ``json`` for a program.
    """
    target = endpoint(url, host, port)
    process = pid()
    render(
        ["url", "pid", "live"],
        [[target, process if process and alive(process) else None, live(target)]],
        output=output,
    )


__all__ = ["alive", "endpoint", "live", "pid", "probe", "start", "status", "stop"]
