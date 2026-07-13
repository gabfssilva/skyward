"""Unit tests for the provisioner's ensure-on-launch (N2) step."""

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

    async def wait(self) -> Any:
        return _Result(0)

    def send_signal(self, signum: int) -> None:
        pass

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        pass


class _Listener:
    _next = 40000

    def __init__(self, remote_port: int) -> None:
        _Listener._next += 1
        self._local = _Listener._next

    def get_port(self) -> int:
        return self._local

    def close(self) -> None:
        pass


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
        self.commands: list[str] = []
        self._file_data = file_data

    async def run(self, command: str, check: bool = False) -> _Result:
        self.commands.append(command)
        return _Result(0)

    async def create_process(self, command: str, **kw: Any) -> _Proc:
        return _Proc()

    async def forward_local_port(
        self, lh: str, lp: int, rh: str, rp: int,
    ) -> _Listener:
        return _Listener(rp)

    def start_sftp_client(self) -> _SFTP:
        return _SFTP(self._file_data)

    def close(self) -> None:
        pass


async def _acoro(v: Any) -> Any:
    return v


class _Resp:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        pass


_NODE = {
    "rank": 0,
    "ip": "1.2.3.4",
    "ssh_port": 22,
    "ssh_user": "root",
    "ssh_key_path": "/keys/id",
    "has_password": False,
}


class _FakeServer:
    """In-memory control plane: sequenced GET /compute responses + POST."""

    def __init__(self, *, get_seq: list[_Resp], post: _Resp | None = None) -> None:
        self._get_seq = get_seq
        self._post = post
        self.get_calls: list[str] = []
        self.post_calls: list[tuple[str, bytes]] = []

    def _next_get(self) -> _Resp:
        if len(self._get_seq) > 1:
            return self._get_seq.pop(0)
        return self._get_seq[0]

    def client(self) -> Any:
        server = self

        class _AC:
            def __init__(self, *a: Any, **k: Any) -> None:
                pass

            async def __aenter__(self) -> _AC:
                return self

            async def __aexit__(self, *a: Any) -> None:
                pass

            async def get(self, url: str) -> _Resp:
                server.get_calls.append(url)
                if url.endswith("/nodes"):
                    return _Resp(200, {"nodes": [_NODE]})
                return server._next_get()

            async def post(self, url: str, **kw: Any) -> _Resp:
                server.post_calls.append((url, kw.get("content", b"")))
                assert server._post is not None
                return server._post

        return _AC


def _ready(nodes: int) -> _Resp:
    return _Resp(200, {"status": "ready", "current_nodes": nodes})


def _not_found() -> _Resp:
    return _Resp(404, {"error": "not found"})


def _accepted() -> _Resp:
    return _Resp(202, {"name": "demo"})


def _conflict() -> _Resp:
    return _Resp(409, {"error": "exists"})


def _make(
    monkeypatch,
    server: _FakeServer,
    *,
    specs: Any = "present",
    poll_interval: float = 0.0,
    poll_timeout: float = 5.0,
) -> tuple[SkywardKernelProvisioner, _Conn]:
    monkeypatch.setattr("httpx.AsyncClient", server.client())
    monkeypatch.setattr("pathlib.Path.exists", lambda self: True)

    if specs == "present":
        monkeypatch.setattr(
            "skyward.config.resolve_pool_specs",
            lambda name, **k: (("SPEC",), "OPTS"),
        )
        monkeypatch.setattr(
            "skyward.server.wire.encode", lambda payload: b"ENCODED",
        )
    else:
        def _missing(name: str, **k: Any) -> Any:
            raise KeyError(f"Pool '{name}' not found. Available: none")

        monkeypatch.setattr("skyward.config.resolve_pool_specs", _missing)

    from jupyter_client.kernelspec import KernelSpec

    conn = _Conn(_remote_conn_file())
    prov = SkywardKernelProvisioner(
        session="demo", url="http://localhost:7590",
        connect_fn=lambda: _acoro(conn),
        kernel_spec=KernelSpec(language="python", argv=["python"]),
    )
    prov.kernel_id = "kid1"
    prov.poll_interval = poll_interval
    prov.poll_timeout = poll_timeout
    return prov, conn


async def test_reuse_ready(monkeypatch):
    server = _FakeServer(get_seq=[_ready(1)])
    prov, _ = _make(monkeypatch, server)
    await prov.pre_launch()
    assert server.post_calls == []
    assert any(u.endswith("/compute/demo") for u in server.get_calls)


async def test_create_then_poll_to_ready(monkeypatch):
    server = _FakeServer(
        get_seq=[_not_found(), _Resp(200, {"status": "provisioning", "current_nodes": 0}), _ready(1)],
        post=_accepted(),
    )
    prov, _ = _make(monkeypatch, server)
    await prov.pre_launch()
    assert len(server.post_calls) == 1
    url, content = server.post_calls[0]
    assert url == "/compute"
    assert content == b"ENCODED"


async def test_race_409_then_poll(monkeypatch):
    server = _FakeServer(
        get_seq=[_not_found(), _ready(1)],
        post=_conflict(),
    )
    prov, _ = _make(monkeypatch, server)
    await prov.pre_launch()
    assert len(server.post_calls) == 1


async def test_no_spec_error_message(monkeypatch):
    server = _FakeServer(get_seq=[_not_found()])
    prov, _ = _make(monkeypatch, server, specs="missing")
    with pytest.raises(RuntimeError, match="sky new demo"):
        await prov.pre_launch()
    assert server.post_calls == []


async def test_slept_pool_error(monkeypatch):
    server = _FakeServer(get_seq=[_ready(0)])
    prov, _ = _make(monkeypatch, server)
    with pytest.raises(RuntimeError, match="scaled to zero"):
        await prov.pre_launch()


async def test_poll_timeout(monkeypatch):
    server = _FakeServer(
        get_seq=[_not_found(), _Resp(200, {"status": "provisioning", "current_nodes": 0})],
        post=_accepted(),
    )
    prov, _ = _make(monkeypatch, server, poll_timeout=0.0)
    with pytest.raises(RuntimeError, match="did not become ready"):
        await prov.pre_launch()
