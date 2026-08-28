"""``sky server`` — run the daemon, or find out whether one is running.

The daemon is the Litestar app at :mod:`skyward.server.http.app` and the process
around it is :mod:`skyward.server.daemon`; this is the command that drives them.
``start`` detaches by default and records the pid, because a control plane that
dies with the terminal that launched it is not a control plane. ``--foreground``
keeps it attached, which is what a dev loop wants.

``stop`` signals the pid rather than asking the daemon to end itself: there is
no shutdown endpoint, and a control plane should not offer one — anything that
can reach the API could then take the whole plane down.

A pool starts a daemon the same way when it finds none (:func:`skyward.core.client.connect`),
so what ``stop`` stops is not only what ``start`` started.
"""

from __future__ import annotations

import asyncio
import os
import signal
import time
from pathlib import Path
from typing import Annotated

import httpx
from cyclopts import Parameter

from skyward.cli import server_app
from skyward.cli._client import HOST, PORT, call, resolve
from skyward.cli._output import Output, render
from skyward.core.client import Client
from skyward.server import daemon
from skyward.shared.schemas import Liveness

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
    while daemon.alive(process):
        if time.monotonic() >= deadline:
            return False
        time.sleep(POLL_SECONDS)
    return True


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
    if not daemon.installed():
        raise SystemExit(daemon.MISSING)

    if foreground:
        daemon.serve(host, port, database)
        return

    if (running := daemon.pid()) and daemon.alive(running):
        raise SystemExit(f"already running (pid {running}) — sky server stop")

    daemon.forget()
    process = daemon.spawn(host, port, database)

    if not _wait_live(f"http://{host}:{port}", timeout):
        if daemon.alive(process):
            os.kill(process, signal.SIGTERM)
        raise SystemExit(f"no answer within {timeout:.0f}s — see {daemon.LOG_FILE}")

    daemon.record(process)
    print(f"http://{host}:{port} (pid {process})")
    print(f"logs: {daemon.LOG_FILE}")


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
    match daemon.pid():
        case None:
            raise SystemExit("no pidfile — nothing to stop")
        case int(process) if not daemon.alive(process):
            daemon.forget()
            print(f"not running (cleared stale pid {process})")
        case int(process):
            os.kill(process, signal.SIGTERM)
            if not _wait_exit(process, timeout):
                raise SystemExit(f"pid {process} still alive after {timeout:.0f}s")
            daemon.forget()
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
    process = daemon.pid()
    render(
        ["url", "pid", "live"],
        [[target, process if process and daemon.alive(process) else None, live(target)]],
        output=output,
    )


__all__ = ["endpoint", "live", "probe", "start", "status", "stop"]
