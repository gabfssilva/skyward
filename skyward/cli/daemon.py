"""sky daemon -- manage the background daemon process."""

from __future__ import annotations

import asyncio

from skyward.cli import daemon_app
from skyward.cli._output import ACTIVE, ERROR, INACTIVE, SUCCESS, console


@daemon_app.command(name="start")
def start() -> None:
    """Start the daemon process (if not already running)."""
    from skyward.daemon.spawn import ensure_daemon, is_daemon_running

    if is_daemon_running():
        console.print(f"{ACTIVE} Daemon already running")
        return
    ensure_daemon()
    console.print(f"{SUCCESS} Daemon started")


@daemon_app.command(name="stop")
def stop() -> None:
    """Stop the daemon process gracefully."""
    from skyward.daemon.client import DaemonClient
    from skyward.daemon.protocol import DaemonStopped, ShutdownDaemon
    from skyward.daemon.spawn import is_daemon_running

    if not is_daemon_running():
        console.print(f"{INACTIVE} Daemon not running")
        return

    async def _stop() -> None:
        async with DaemonClient() as client:
            resp = await client.request(ShutdownDaemon())
            match resp:
                case DaemonStopped():
                    console.print(f"{SUCCESS} Daemon stopped")
                case _:
                    console.print(f"{ERROR} Unexpected response: {resp}")

    try:
        asyncio.run(_stop())
    except ConnectionError:
        console.print(f"{ERROR} Could not connect to daemon")


@daemon_app.command(name="status")
def status() -> None:
    """Check if the daemon is running."""
    from skyward.daemon.spawn import is_daemon_running

    if is_daemon_running():
        console.print(f"{ACTIVE} Daemon running")
    else:
        console.print(f"{INACTIVE} Daemon not running")
