"""``sky server`` — run the daemon, or find out whether one is running.

The daemon is the Litestar app at :mod:`skyward.server.http.app` and the process
around it is :mod:`skyward.server.daemon`; this is the command that drives them.
``start`` detaches by default and records the pid, because a control plane that
dies with the terminal that launched it is not a control plane. ``--foreground``
keeps it attached, which is what a dev loop wants.

``stop`` signals the pid rather than asking the daemon to end itself: there is
no shutdown endpoint, and a control plane should not offer one — anything that
can reach the API could then take the whole plane down. ``restart`` is the two of
them in that order, and skips the stop when there is nothing running.

A pool starts a daemon the same way when it finds none (:func:`skyward.core.client.connect`),
so what ``stop`` stops is not only what ``start`` started.

``interface set`` writes down a network interface for every daemon started here to
listen on beside its host. It restarts nothing: the daemon running holds live
computes, and when it stops is its owner's call.
"""

from __future__ import annotations

import asyncio
import os
import signal
import time
from pathlib import Path
from typing import Annotated

import httpx
from cyclopts import App, Parameter

from skyward.api.v1 import LivenessResource
from skyward.cli import server_app
from skyward.cli._client import HOST, PORT, call, resolve
from skyward.cli._output import Output, render
from skyward.core.client import Client
from skyward.server import daemon
from skyward.shared.observability import LogLevel, notice

POLL_SECONDS = 0.2

interface_app = App(name="interface", help="Choose the network interface the daemon listens on")
server_app.command(interface_app)


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
        return (await client.call("GET", "/v1/health/live", LivenessResource)).live
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
    log_level: Annotated[LogLevel | None, Parameter(help="Console verbosity (the log file always takes DEBUG)")] = None,
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
    log_level
        How much the daemon says on its console. ``DEBUG`` is every decision the
        control plane takes; the file under ``~/.skyward/logs`` gets that either way.
    """
    if not daemon.installed():
        raise SystemExit(daemon.MISSING)

    if foreground:
        daemon.serve(host, port, database, log_level)
        return

    if (running := daemon.pid()) and daemon.alive(running):
        raise SystemExit(f"already running (pid {running}) — sky server stop")

    daemon.forget()
    process = daemon.spawn(host, port, database, log_level)

    if not _wait_live(f"http://{host}:{port}", timeout):
        if daemon.alive(process):
            os.kill(process, signal.SIGTERM)
        raise SystemExit(f"no answer within {timeout:.0f}s — see {daemon.LOG_FILE}")

    daemon.record(process)
    print(f"http://{host}:{port} (pid {process})")
    for address in daemon.listening(host):
        if address != host:
            target = f"http://{address}:{port}"
            if live(target):
                print(target)
            else:
                notice("WARNING", f"nothing answers at {target} — see {daemon.LOG_FILE}")
    if (name := daemon.interface()) and not daemon.addresses(name):
        notice("WARNING", f"{name} has no IPv4 address, so the daemon listens on {host} alone")
    print(f"logs: {daemon.LOG_FILE}")


def _halt(timeout: float) -> str | None:
    """Stop the daemon this machine recorded and say what happened, or None when it had recorded none."""
    match daemon.pid():
        case None:
            return None
        case int(process) if not daemon.alive(process):
            daemon.forget()
            return f"not running (cleared stale pid {process})"
        case int(process):
            os.kill(process, signal.SIGTERM)
            if not _wait_exit(process, timeout):
                raise SystemExit(f"pid {process} still alive after {timeout:.0f}s")
            daemon.forget()
            return f"stopped (pid {process})"


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
    match _halt(timeout):
        case None:
            raise SystemExit("no pidfile — nothing to stop")
        case str(said):
            print(said)


@server_app.command(name="restart")
def restart(
    *,
    host: Annotated[str, Parameter(help="Bind address")] = HOST,
    port: Annotated[int, Parameter(help="Bind port")] = PORT,
    timeout: Annotated[float, Parameter(help="Seconds to wait for the old one to leave, then for the new one to answer")] = 30.0,
    database: Annotated[Path | None, Parameter(help="SQLite path (default: ~/.skyward/skyward.sqlite)")] = None,
    log_level: Annotated[LogLevel | None, Parameter(help="Console verbosity (the log file always takes DEBUG)")] = None,
) -> None:
    """Stop the daemon this machine started and start one in its place.

    The machines outlive it. A compute belongs to the daemon rather than to the
    process, so what a restart costs is a gap in reconciliation and every stream
    that was open — clients come back on their own cursor. A compute whose client
    is renewing its lease is not at risk from a restart that takes a second; one
    nobody has held for a minute was already on its way out.

    Nothing to stop is not a failure here: a daemon that died on its own is
    restarted by the same command that restarts a live one.

    Parameters
    ----------
    host
        Address to bind.
    port
        Port to bind.
    timeout
        How long to wait for the old process to leave, and then for the new one to answer.
    database
        The SQLite file the daemon keeps its state in.
    log_level
        How much the daemon says on its console. ``DEBUG`` is every decision the
        control plane takes; the file under ``~/.skyward/logs`` gets that either way.
    """
    if said := _halt(timeout):
        print(said)
    elif live(f"http://{host}:{port}"):
        raise SystemExit(f"something answers at http://{host}:{port} and this machine recorded no pid for it — stop it where it was started")

    start(host=host, port=port, timeout=timeout, database=database, log_level=log_level)


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


@interface_app.command(name="set")
def set_interface(
    interface: Annotated[str, Parameter(help="A network interface (en0, tailscale0), one of its IPv4 addresses, or 0.0.0.0 for all of them")],
) -> None:
    """Make the daemon listen on a network interface too, from its next start.

    It goes on listening where it did, which is where this machine's pools and
    commands look for it; the interface is where the rest of the network reaches
    it. A name is read for its IPv4 addresses whenever a daemon starts, so one the
    interface was given since is the one it listens on.

    The daemon running now is left as it is, with every compute it holds:
    ``sky server restart`` is what applies it.

    Parameters
    ----------
    interface
        The interface's name, an address this machine holds, or ``0.0.0.0``.
    """
    if not daemon.installed():
        raise SystemExit(daemon.MISSING)

    known = daemon.interfaces()
    held = {address for addresses in known.values() for address in addresses}
    if interface not in known and interface not in held and interface != daemon.WILDCARD:
        named = ", ".join(sorted(name for name, addresses in known.items() if addresses))
        raise SystemExit(f"this machine has no interface or address '{interface}'; the ones with an address: {named}")

    daemon.choose(interface)
    reached = daemon.addresses(interface)
    print(f"{interface} ({', '.join(reached) or 'no IPv4 address yet'}), from the daemon's next start: sky server restart")


__all__ = ["endpoint", "interface_app", "live", "probe", "restart", "set_interface", "start", "status", "stop"]
