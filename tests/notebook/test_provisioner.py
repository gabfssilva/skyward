"""Unit tests for the Skyward remote-kernel provisioner."""

from __future__ import annotations

import json
from typing import Any

import pytest

from skyward.notebook.provisioner import SkywardKernelProvisioner

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


REMOTE_KEY = "abc123deadbeef"
REMOTE_PORTS = {
    "shell_port": 51001,
    "iopub_port": 51002,
    "stdin_port": 51003,
    "control_port": 51004,
    "hb_port": 51005,
}


def _remote_conn_file() -> bytes:
    return json.dumps({
        "ip": "127.0.0.1",
        "transport": "tcp",
        "signature_scheme": "hmac-sha256",
        "key": REMOTE_KEY,
        **REMOTE_PORTS,
    }).encode()


class _Result:
    def __init__(self, code: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.exit_status = code
        self.stdout = stdout
        self.stderr = stderr


class _Proc:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.signals: list[int] = []
        self.terminated = False
        self.killed = False

    async def wait(self) -> Any:
        self.returncode = 0
        return _Result(0)

    def send_signal(self, signum: int) -> None:
        self.signals.append(signum)

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


class _Listener:
    _next = 40000

    def __init__(self, remote_port: int) -> None:
        self.remote_port = remote_port
        _Listener._next += 1
        self._local = _Listener._next
        self.closed = False

    def get_port(self) -> int:
        return self._local

    def close(self) -> None:
        self.closed = True


class _SFTPFile:
    def __init__(self, data: bytes) -> None:
        self._data = data

    async def read(self) -> bytes:
        return self._data

    async def __aenter__(self) -> _SFTPFile:
        return self

    async def __aexit__(self, *a: Any) -> None:
        pass


class _SFTP:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def open(self, path: str, mode: str) -> _SFTPFile:
        return _SFTPFile(self._data)

    async def __aenter__(self) -> _SFTP:
        return self

    async def __aexit__(self, *a: Any) -> None:
        pass


class _Conn:
    def __init__(self, file_data: bytes) -> None:
        self.forwards: list[tuple[str, int, str, int]] = []
        self.listeners: list[_Listener] = []
        self.commands: list[str] = []
        self.procs: list[_Proc] = []
        self.closed = False
        self._file_data = file_data

    async def run(self, command: str, check: bool = False) -> _Result:
        self.commands.append(command)
        return _Result(0)

    async def create_process(self, command: str, **kw: Any) -> _Proc:
        self.commands.append(command)
        proc = _Proc()
        self.procs.append(proc)
        return proc

    async def forward_local_port(
        self, lh: str, lp: int, rh: str, rp: int,
    ) -> _Listener:
        self.forwards.append((lh, lp, rh, rp))
        lis = _Listener(rp)
        self.listeners.append(lis)
        return lis

    def start_sftp_client(self) -> _SFTP:
        return _SFTP(self._file_data)

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        pass


def _make(
    monkeypatch,
    *,
    ssh_user: str = "root",
    has_password: bool = False,
    key_exists: bool = True,
    conn: _Conn | None = None,
) -> tuple[SkywardKernelProvisioner, _Conn]:
    conn = conn or _Conn(_remote_conn_file())

    node = {
        "rank": 0,
        "ip": "1.2.3.4",
        "ssh_port": 22,
        "ssh_user": ssh_user,
        "ssh_key_path": "/keys/id" if key_exists else None,
        "has_password": has_password,
    }

    class _Resp:
        def __init__(self, payload: dict) -> None:
            self.status_code = 200
            self._payload = payload

        def json(self) -> dict:
            return self._payload

        def raise_for_status(self) -> None:
            pass

    class _AC:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        async def __aenter__(self) -> _AC:
            return self

        async def __aexit__(self, *a: Any) -> None:
            pass

        async def get(self, url: str) -> _Resp:
            if url.endswith("/nodes"):
                return _Resp({"name": "demo", "nodes": [node]})
            return _Resp({"status": "ready", "current_nodes": 1})

    monkeypatch.setattr("httpx.AsyncClient", _AC)
    monkeypatch.setattr("pathlib.Path.exists", lambda self: key_exists)

    from jupyter_client.kernelspec import KernelSpec

    prov = SkywardKernelProvisioner(
        session="demo", url="http://localhost:7590",
        connect_fn=lambda: _acoro(conn),
        kernel_spec=KernelSpec(language="python", argv=["python"]),
    )
    prov.kernel_id = "kid1"
    return prov, conn


async def _acoro(v: Any) -> Any:
    return v


async def _run(prov: SkywardKernelProvisioner) -> Any:
    kwargs = await prov.pre_launch()
    cmd = kwargs.pop("cmd")
    info = await prov.launch_kernel(cmd, **kwargs)
    kwargs["cmd"] = cmd
    return kwargs, info


async def test_five_forwards_on_one_connection(monkeypatch):
    prov, conn = _make(monkeypatch)
    await _run(prov)
    assert len(conn.forwards) == 5
    remote_forwarded = {rp for _, _, _, rp in conn.forwards}
    assert remote_forwarded == set(REMOTE_PORTS.values())


async def test_launch_cmd_targets_venv_python_fresh_path(monkeypatch):
    prov, conn = _make(monkeypatch)
    kwargs, _ = await _run(prov)
    cmd = " ".join(kwargs["cmd"])
    assert "/opt/skyward/.venv/bin/python -m ipykernel_launcher" in cmd
    assert "/opt/skyward/kernel-kid1.json" in cmd
    assert "cd /opt/skyward" in cmd


async def test_sudo_wrapping_when_not_root(monkeypatch):
    prov, conn = _make(monkeypatch, ssh_user="ubuntu")
    kwargs, _ = await _run(prov)
    cmd = " ".join(kwargs["cmd"])
    assert cmd.startswith("sudo bash -c ")


async def test_no_sudo_when_root(monkeypatch):
    prov, conn = _make(monkeypatch, ssh_user="root")
    kwargs, _ = await _run(prov)
    assert not " ".join(kwargs["cmd"]).startswith("sudo")


async def test_connection_info_carries_remote_key_and_local_ports(monkeypatch):
    prov, conn = _make(monkeypatch)
    _, info = await _run(prov)
    assert info["ip"] == "127.0.0.1"
    assert info["key"] == REMOTE_KEY.encode()
    assert info["transport"] == "tcp"
    assert info["signature_scheme"] == "hmac-sha256"
    locals_ = {lis.get_port() for lis in conn.listeners}
    for channel in REMOTE_PORTS:
        assert info[channel] in locals_


async def test_cmd_present_after_pre_launch(monkeypatch):
    prov, conn = _make(monkeypatch)
    kwargs = await prov.pre_launch()
    assert isinstance(kwargs["cmd"], list)
    assert kwargs["cmd"]


async def test_lifecycle_maps_to_proc(monkeypatch):
    prov, conn = _make(monkeypatch)
    await _run(prov)
    assert prov.has_process
    assert await prov.poll() is None
    await prov.send_signal(2)
    assert conn.procs[0].signals == [2]
    await prov.terminate()
    assert conn.procs[0].terminated
    await prov.kill()
    assert conn.procs[0].killed
    assert await prov.wait() == 0


async def test_ensures_ipykernel(monkeypatch):
    prov, conn = _make(monkeypatch)
    await prov.pre_launch()
    assert any("add ipykernel" in c and "uv" in c for c in conn.commands)


async def test_restart_reuses_connection_and_reforwards(monkeypatch):
    prov, conn = _make(monkeypatch)
    await _run(prov)
    first_listeners = list(conn.listeners)
    assert len(first_listeners) == 5

    await prov.cleanup(restart=True)
    assert not conn.closed
    assert all(lis.closed for lis in first_listeners)

    await _run(prov)
    assert not conn.closed
    new_listeners = conn.listeners[5:]
    assert len(new_listeners) == 5
    assert all(not lis.closed for lis in new_listeners)


async def test_cleanup_no_restart_closes_connection(monkeypatch):
    prov, conn = _make(monkeypatch)
    await _run(prov)
    await prov.cleanup(restart=False)
    assert conn.closed
    assert all(lis.closed for lis in conn.listeners)


async def test_missing_key_refused(monkeypatch):
    prov, conn = _make(monkeypatch, key_exists=False)
    with pytest.raises(RuntimeError, match="same host"):
        await prov.pre_launch()


async def test_password_only_refused(monkeypatch):
    prov, conn = _make(monkeypatch, key_exists=False, has_password=True)
    with pytest.raises(RuntimeError, match="password-only"):
        await prov.pre_launch()
