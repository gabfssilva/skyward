"""Standalone asyncssh helpers for the remote-kernel provisioner.

A small, Casty-free wrapper around the asyncssh recipes used by the
interactive PTY (:mod:`skyward.cli._interactive`) and the SSH transport
actor: connect with an injectable ``connect_fn`` (for tests), open local
port-forwards, run remote commands, and SFTP-read a remote file with
bounded polling until it exists.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any


def default_connect(
    host: str, port: int, user: str, key_path: str,
) -> Callable[[], Awaitable[Any]]:
    """Return a zero-arg coroutine factory that opens an asyncssh connection.

    Parameters
    ----------
    host, port, user, key_path
        SSH coordinates and the local private-key path.

    Returns
    -------
    Callable[[], Awaitable[Any]]
        A factory suitable for injection wherever a ``connect_fn`` is expected.
    """
    import asyncssh

    return lambda: asyncssh.connect(
        host, port=port, username=user,
        client_keys=[key_path] if key_path else [],
        known_hosts=None,
    )


async def run(conn: Any, command: str) -> tuple[int, str, str]:
    """Run a command over SSH and return ``(exit_code, stdout, stderr)``."""
    result = await conn.run(command, check=False)
    return result.exit_status or 0, str(result.stdout or ""), str(result.stderr or "")


async def forward_local_port(conn: Any, remote_port: int) -> tuple[int, Any]:
    """Forward an ephemeral local port to ``127.0.0.1:remote_port`` on the node.

    Returns
    -------
    tuple[int, Any]
        The allocated local port and the listener object.
    """
    listener = await conn.forward_local_port("127.0.0.1", 0, "127.0.0.1", remote_port)
    return listener.get_port(), listener


async def read_remote_file(
    conn: Any, remote_path: str, *, attempts: int = 60, delay: float = 0.5,
) -> bytes:
    """SFTP-read a remote file, polling until it exists.

    Parameters
    ----------
    conn
        An open asyncssh connection.
    remote_path
        Absolute path on the node.
    attempts
        Maximum number of poll attempts before giving up.
    delay
        Seconds between attempts.

    Returns
    -------
    bytes
        The file contents.

    Raises
    ------
    TimeoutError
        If the file does not appear within ``attempts * delay`` seconds.
    """
    last: Exception | None = None
    try:
        async with asyncio.timeout(attempts * delay):
            while True:
                try:
                    async with conn.start_sftp_client() as sftp, sftp.open(remote_path, "rb") as f:
                        return await f.read()
                except Exception as exc:  # noqa: BLE001
                    last = exc
                    await asyncio.sleep(delay)
    except TimeoutError:
        raise TimeoutError(f"remote file {remote_path!r} did not appear") from last
