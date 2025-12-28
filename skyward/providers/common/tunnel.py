"""Shared tunnel utilities for all providers."""

from __future__ import annotations

import socket
from subprocess import PIPE, Popen

from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_delay,
    wait_fixed,
)

RPYC_PORT = 18861


class TunnelNotReadyError(Exception):
    """Tunnel not ready - retry."""


def find_available_port() -> int:
    """Find an available local port."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


def wait_for_tunnel(port: int, timeout: int = 30) -> None:
    """Wait for tunnel to accept connections.

    Retries TCP connection to localhost:port until successful or timeout.

    Args:
        port: Local port to check.
        timeout: Maximum time to wait in seconds.

    Raises:
        TimeoutError: If tunnel doesn't become ready within timeout.
    """

    @retry(
        stop=stop_after_delay(timeout),
        wait=wait_fixed(0.5),
        retry=retry_if_exception_type(TunnelNotReadyError),
        reraise=True,
    )
    def _check() -> None:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except (TimeoutError, ConnectionRefusedError, OSError):
            raise TunnelNotReadyError() from None

    try:
        _check()
    except RetryError as e:
        raise TimeoutError(f"Tunnel not ready on port {port}") from e


def create_tunnel(
    cmd: list[str],
    local_port: int,
    timeout: int = 30,
) -> tuple[int, Popen[bytes]]:
    """Create a tunnel process and wait for it to be ready.

    Args:
        cmd: Command to run for the tunnel subprocess.
        local_port: Local port the tunnel should listen on.
        timeout: Timeout in seconds to wait for tunnel.

    Returns:
        Tuple of (local_port, process).

    Raises:
        TimeoutError: If tunnel doesn't become ready within timeout.
    """
    import subprocess

    proc = subprocess.Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)

    try:
        wait_for_tunnel(local_port, timeout=timeout)
        return local_port, proc
    except Exception:
        proc.terminate()
        raise
