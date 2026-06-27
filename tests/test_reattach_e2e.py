"""End-to-end reattach: server death does not leak instances; restart re-adopts.

Spawns a real ``sky server`` daemon (isolated ``HOME`` so handles/PID/TLS
live in a tmp dir), creates a container-backed session, SIGKILLs the server
(leaving the container alive), restarts, and asserts the pool is re-adopted
onto the *same* container — proving the handle persists, reattach skips
re-provisioning, and ``stop`` removes the handle.

Requires Docker (container provider). Run via ``task test:e2e``.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

import skyward as sky
from skyward.api.spec import Options, Spec
from skyward.providers import Container
from skyward.server.wire import decode, encode

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(420), pytest.mark.xdist_group("reattach")]

_PORT = 7599


@sky.function
def _hostname() -> str:
    import socket

    return socket.gethostname()


def _spawn_server(home: Path, port: int) -> subprocess.Popen:
    home.mkdir(parents=True, exist_ok=True)
    # The container provider reads ~/.ssh/id_* — link the real key into the
    # isolated HOME so handles/PID/TLS stay in tmp without losing SSH auth.
    ssh_link = home / ".ssh"
    if not ssh_link.exists():
        ssh_link.symlink_to(Path.home() / ".ssh")
    log = (home / "server.log").open("ab")  # noqa: SIM115 — closed when the child exits
    env = {**os.environ, "HOME": str(home), "SKYWARD_SERVER_LOG": str(home / "skyward.log")}
    return subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "skyward.server:create_app", "--factory",
            "--host", "127.0.0.1", "--port", str(port),
        ],
        stdout=log, stderr=log, stdin=subprocess.DEVNULL,
        start_new_session=True, env=env,
    )


def _wait_health(url: str, *, timeout: float = 90.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{url}/health", timeout=1.0).status_code == 200:
                return True
        except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException):
            pass
        time.sleep(0.3)
    return False


def _wait_ready(url: str, name: str, *, timeout: float = 240.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = httpx.get(f"{url}/compute/{name}", timeout=5.0)
        if r.status_code == 200 and r.json()["status"] == "ready":
            return
        if r.status_code == 200 and r.json()["status"] == "failed":
            raise AssertionError(f"pool failed: {r.json()}")
        time.sleep(1.0)
    raise AssertionError("pool did not become ready in time")


def _run_remote(url: str, name: str) -> str:
    body = encode(_hostname())
    r = httpx.post(
        f"{url}/compute/{name}/executions", params={"mode": "run"},
        content=body, headers={"Content-Type": "application/octet-stream"}, timeout=30.0,
    )
    assert r.status_code == 202, r.text
    eid = r.json()["id"]
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        got = httpx.get(f"{url}/compute/{name}/executions/{eid}", timeout=30.0)
        if got.status_code == 200:
            return decode(got.content)
        if got.status_code == 500:
            raise AssertionError(f"execution error: {decode(got.content)!r}")
        time.sleep(0.5)
    raise AssertionError("execution did not finish")


def _kill(proc: subprocess.Popen) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.kill(proc.pid, signal.SIGKILL)
    proc.wait(timeout=10)


def test_server_restart_reattaches_same_container(tmp_path: Path) -> None:
    home = tmp_path / "home"
    url = f"http://127.0.0.1:{_PORT}"
    name = "reattach-e2e"
    sessions = home / ".skyward" / "sessions"

    server = _spawn_server(home, _PORT)
    try:
        assert _wait_health(url), "server did not become healthy"

        body = encode(((Spec(provider=Container()),), Options()))
        r = httpx.post(
            f"{url}/compute", params={"name": name}, content=body,
            headers={"Content-Type": "application/octet-stream"}, timeout=30.0,
        )
        assert r.status_code == 202, r.text
        _wait_ready(url, name)

        assert (sessions / f"{name}.json").exists(), "handle was not persisted"
        host_before = _run_remote(url, name)

        # crash the server, leaving the container alive
        _kill(server)
        server = _spawn_server(home, _PORT)
        assert _wait_health(url), "server did not restart"

        # reattach should repopulate the pool over the same container
        info = httpx.get(f"{url}/compute/{name}", timeout=10.0)
        assert info.status_code == 200, info.text
        assert info.json()["current_nodes"] == 1
        host_after = _run_remote(url, name)
        assert host_after == host_before, "reattached onto a different container"

        # stop tears down and removes the handle
        assert httpx.delete(f"{url}/compute/{name}", timeout=60.0).status_code == 204
        assert not (sessions / f"{name}.json").exists(), "handle survived stop"
    finally:
        _kill(server)
