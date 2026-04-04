"""Auto-spawn the daemon process."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from pathlib import Path

from skyward.observability.logger import logger

log = logger.bind(component="daemon-spawn")

_DEFAULT_SOCKET = Path.home() / ".skyward" / "daemon.sock"


def daemon_socket_path() -> Path:
    return _DEFAULT_SOCKET


def is_daemon_running(socket_path: Path = _DEFAULT_SOCKET) -> bool:
    """Check if the daemon is listening on the socket."""
    if not socket_path.exists():
        return False
    try:
        loop = asyncio.new_event_loop()
        try:
            reader, writer = loop.run_until_complete(
                asyncio.open_unix_connection(str(socket_path)),
            )
            writer.close()
            loop.run_until_complete(writer.wait_closed())
            return True
        except (ConnectionRefusedError, FileNotFoundError, OSError):
            return False
        finally:
            loop.close()
    except Exception:
        return False


def ensure_daemon(socket_path: Path = _DEFAULT_SOCKET, timeout: float = 10.0) -> None:
    """Start the daemon if not already running, wait until ready."""
    if is_daemon_running(socket_path):
        return

    log.info("Starting daemon process...")
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = socket_path.parent / "daemon.log"

    with log_path.open("a") as log_file:
        subprocess.Popen(
            [sys.executable, "-m", "skyward.daemon"],
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_daemon_running(socket_path):
            log.info("Daemon ready")
            return
        time.sleep(0.2)

    raise RuntimeError(f"Daemon failed to start within {timeout}s. Check {log_path}")
