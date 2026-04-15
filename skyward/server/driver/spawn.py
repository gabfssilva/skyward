"""Spawn-or-connect helper for the long-lived server process.

The client calls :func:`ensure_server` to get a live UDS path. If no
server is running, it spawns ``python -m skyward.server.host`` as a
detached subprocess, waits for ``/v1/health`` to answer, and returns
the path. A filesystem lock serializes concurrent callers so the
process is created exactly once.
"""
from __future__ import annotations

import asyncio
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

_DEFAULT_DIR = Path.home() / ".skyward"
_DEFAULT_SOCKET = _DEFAULT_DIR / "server.sock"
_DEFAULT_LOCK = _DEFAULT_DIR / "server.lock"


async def _probe(socket_path: str, timeout: float = 1.0) -> bool:
    """Return ``True`` if ``/v1/health`` answers ``200`` on the socket."""
    try:
        transport = httpx.AsyncHTTPTransport(uds=socket_path)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://skyward", timeout=timeout,
            headers={"X-Skyward-Api": "1"},
        ) as client:
            resp = await client.get("/v1/health")
            return resp.status_code == 200
    except Exception:
        return False


def _acquire_lock(lock_path: Path) -> Any:
    """Best-effort advisory flock; returns the open fd on success.

    The fd is intentionally leaked to the caller's scope — closing it
    would drop the lock. Callers hand the fd back via :func:`_release_lock`.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def _release_lock(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _spawn(socket_path: Path) -> None:
    """Detach the host process into a new session."""
    subprocess.Popen(
        [sys.executable, "-m", "skyward.server.host", "--socket", str(socket_path)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )


async def ensure_server(
    socket_path: str | os.PathLike[str] = _DEFAULT_SOCKET,
    lock_path: str | os.PathLike[str] = _DEFAULT_LOCK,
    *,
    deadline_s: float = 10.0,
    backoff_s: float = 0.1,
) -> str:
    """Return a live host UDS path, spawning the server if needed.

    Parameters
    ----------
    socket_path
        Preferred UDS location. Defaults to ``~/.skyward/server.sock``.
    lock_path
        Advisory flock used to serialize concurrent ``ensure_server``
        callers. Defaults to ``~/.skyward/server.lock``.
    deadline_s
        Maximum seconds to wait for the server to come up.
    backoff_s
        Initial health-probe interval; doubles up to 1 s.

    Returns
    -------
    str
        Absolute path of the UDS the host is accepting on.

    Raises
    ------
    TimeoutError
        If the server does not answer ``/v1/health`` within
        ``deadline_s`` seconds after being spawned.
    """
    sock = Path(socket_path)
    lock = Path(lock_path)

    if await _probe(str(sock)):
        return str(sock)

    fd = _acquire_lock(lock)
    try:
        if await _probe(str(sock)):
            return str(sock)

        _spawn(sock)
        deadline = time.monotonic() + deadline_s
        delay = backoff_s
        while time.monotonic() < deadline:
            if await _probe(str(sock)):
                return str(sock)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 1.0)
        raise TimeoutError(
            f"server did not become healthy within {deadline_s}s",
        )
    finally:
        _release_lock(fd)


__all__ = ["ensure_server"]
