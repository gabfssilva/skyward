"""Tests for sky console / sky repl and the PTY driver."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from skyward.cli import app
from skyward.cli._interactive import _on_resize, open_pty

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


# ── PTY driver ───────────────────────────────────────────────────


class _Stdout:
    def __init__(self, chunks: list[bytes]) -> None:
        self._it = iter(chunks)

    async def read(self, _n: int) -> bytes:
        try:
            return next(self._it)
        except StopIteration:
            return b""


class _Proc:
    def __init__(self, chunks: list[bytes], exit_status: int = 0) -> None:
        self.stdout = _Stdout(chunks)
        self.stdin = MagicMock()
        self._exit = exit_status

    async def wait(self) -> SimpleNamespace:
        return SimpleNamespace(exit_status=self._exit)


class _Conn:
    def __init__(self, proc: _Proc) -> None:
        self._proc = proc
        self.created: dict[str, Any] = {}

    async def create_process(self, command, **kw):  # noqa: ANN001, ANN003
        self.created["command"] = command
        self.created["kw"] = kw
        return self._proc

    def close(self) -> None:
        pass

    async def wait_closed(self) -> None:
        pass


async def _acoro(v: Any) -> Any:
    return v


async def test_open_pty_requests_pty_and_pumps(capsysbinary):
    conn = _Conn(_Proc([b"hello", b""]))
    code = await open_pty("h", 2200, "u", "/k", None, None, connect_fn=lambda: _acoro(conn))
    assert code == 0
    assert conn.created["command"] is None
    assert conn.created["kw"]["term_type"]
    assert isinstance(conn.created["kw"]["term_size"], tuple)
    assert b"hello" in capsysbinary.readouterr().out


async def test_open_pty_returns_exit_status(capsysbinary):
    conn = _Conn(_Proc([b""], exit_status=3))
    code = await open_pty("h", 22, "u", "/k", None, "python", connect_fn=lambda: _acoro(conn))
    assert code == 3
    assert conn.created["command"] == "python"


def test_on_resize_calls_change_terminal_size():
    proc = MagicMock()
    _on_resize(proc)
    proc.change_terminal_size.assert_called_once()
    cols, rows = proc.change_terminal_size.call_args.args
    assert cols > 0
    assert rows > 0


# ── CLI wiring ───────────────────────────────────────────────────


@pytest.fixture
def keyfile(tmp_path):
    f = tmp_path / "id_key"
    f.write_text("PRIVATE KEY")
    return str(f)


@pytest.fixture
def stub(monkeypatch):
    def install(response: httpx.Response):
        class _T(httpx.BaseTransport):
            def handle_request(self, request: httpx.Request) -> httpx.Response:
                return response

        monkeypatch.setattr(
            "skyward.cli.interactive.make_client",
            lambda url: httpx.Client(base_url=url or "http://localhost:7590", transport=_T()),
        )

    return install


def _nodes(keyfile: str, *, has_password: bool = False, ip: str = "1.2.3.4") -> httpx.Response:
    return httpx.Response(200, json={"name": "demo", "nodes": [
        {"rank": 0, "is_head": True, "status": "ready", "instance_id": "i-0", "ip": ip,
         "ssh_port": 22, "ssh_user": "ubuntu", "ssh_key_path": keyfile, "has_password": has_password},
        {"rank": 1, "is_head": False, "status": "ready", "instance_id": "i-1", "ip": "5.6.7.8",
         "ssh_port": 22, "ssh_user": "ubuntu", "ssh_key_path": keyfile, "has_password": False},
    ]})


def test_console_opens_head(stub, keyfile, monkeypatch):
    captured: dict[str, Any] = {}

    async def fake_open_pty(host, port, user, key, pw, command, env=None, **kw):  # noqa: ANN001, ANN003
        captured.update(host=host, port=port, command=command)
        return 0

    stub(_nodes(keyfile))
    monkeypatch.setattr("skyward.cli.interactive.open_pty", fake_open_pty)
    with pytest.raises(SystemExit) as e:
        app(["console", "demo"], exit_on_error=False)
    assert e.value.code == 0
    assert captured == {"host": "1.2.3.4", "port": 22, "command": None}


def test_repl_uses_python_command(stub, keyfile, monkeypatch):
    captured: dict[str, Any] = {}

    async def fake_open_pty(host, port, user, key, pw, command, env=None, **kw):  # noqa: ANN001, ANN003
        captured["command"] = command
        return 0

    stub(_nodes(keyfile))
    monkeypatch.setattr("skyward.cli.interactive.open_pty", fake_open_pty)
    with pytest.raises(SystemExit) as e:
        app(["repl", "demo"], exit_on_error=False)
    assert e.value.code == 0
    assert "/opt/skyward/.venv/bin/python" in captured["command"]


def test_console_selects_rank(stub, keyfile, monkeypatch):
    captured: dict[str, Any] = {}

    async def fake_open_pty(host, *a, **kw):  # noqa: ANN002, ANN003
        captured["host"] = host
        return 0

    stub(_nodes(keyfile))
    monkeypatch.setattr("skyward.cli.interactive.open_pty", fake_open_pty)
    with pytest.raises(SystemExit):
        app(["console", "demo", "--rank", "1"], exit_on_error=False)
    assert captured["host"] == "5.6.7.8"


def test_console_propagates_exit(stub, keyfile, monkeypatch):
    async def fake_open_pty(*a, **kw):  # noqa: ANN002, ANN003
        return 42

    stub(_nodes(keyfile))
    monkeypatch.setattr("skyward.cli.interactive.open_pty", fake_open_pty)
    with pytest.raises(SystemExit) as e:
        app(["console", "demo"], exit_on_error=False)
    assert e.value.code == 42


def test_console_missing_rank_errors(stub, keyfile):
    stub(_nodes(keyfile))
    with pytest.raises(SystemExit) as e:
        app(["console", "demo", "--rank", "9"], exit_on_error=False)
    assert e.value.code == 1


def test_console_missing_key_errors(stub):
    stub(_nodes("/nonexistent/key/path"))
    with pytest.raises(SystemExit) as e:
        app(["console", "demo"], exit_on_error=False)
    assert e.value.code == 1


def test_console_password_only_errors(stub, capsys):
    stub(_nodes("/nonexistent/key", has_password=True))
    with pytest.raises(SystemExit) as e:
        app(["console", "demo"], exit_on_error=False)
    assert e.value.code == 1
    assert "password-only" in capsys.readouterr().out
