"""Tests for sky install (imperative uv add)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from skyward.cli import app

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class _StubTransport(httpx.BaseTransport):
    def __init__(self, responder) -> None:
        self.responder = responder
        self.calls: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        return self.responder(request)


@pytest.fixture
def stub(monkeypatch):
    captured: dict[str, Any] = {}

    def install_(responder) -> _StubTransport:
        transport = _StubTransport(responder)
        captured["transport"] = transport
        monkeypatch.setattr(
            "skyward.cli.compute.make_client",
            lambda url: httpx.Client(base_url=url or "http://localhost:7590", transport=transport),
        )
        return transport

    return install_


def test_build_install_pending_is_pending_function():
    from skyward.api.function import PendingFunction
    from skyward.cli._install import build_install_pending

    pending = build_install_pending(("numpy",))
    assert isinstance(pending, PendingFunction)


def test_uv_add_invokes_resolved_uv(monkeypatch):
    from skyward.cli._install import build_install_pending

    calls: dict[str, Any] = {}

    class _FakeProc:
        stdout = iter(["building...\n"])

        def wait(self):
            return 0

    def fake_popen(cmd, **kw):
        calls["cmd"] = cmd
        calls["kw"] = kw
        return _FakeProc()

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/uv")

    pending = build_install_pending(("numpy", "torch==2.3"))
    result = pending.fn(*pending.args, **pending.kwargs)

    assert result == {"exit": 0}
    assert calls["cmd"] == ["/usr/bin/uv", "add", "numpy", "torch==2.3"]
    assert calls["kw"]["cwd"] == "/opt/skyward"


def test_uv_add_falls_back_to_local_bin(monkeypatch):
    from skyward.cli._install import build_install_pending

    seen: dict[str, Any] = {}

    class _FakeProc:
        stdout = iter([])

        def wait(self):
            return 7

    def fake_popen(cmd, **kw):
        seen["cmd"] = cmd
        return _FakeProc()

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    monkeypatch.setattr("shutil.which", lambda _name: None)

    pending = build_install_pending(("x",))
    result = pending.fn(*pending.args, **pending.kwargs)
    assert result == {"exit": 7}
    assert seen["cmd"][0] == "/root/.local/bin/uv"


def test_resolve_specifiers_merges(tmp_path):
    from skyward.cli.compute import _resolve_specifiers

    req = tmp_path / "r.txt"
    req.write_text("# comment\nnumpy\n\npandas==2.0\n")
    assert _resolve_specifiers(["torch"], req) == ("torch", "numpy", "pandas==2.0")


def test_resolve_specifiers_empty_errors():
    from skyward.cli.compute import _resolve_specifiers

    with pytest.raises(SystemExit):
        _resolve_specifiers(None, None)


def test_resolve_specifiers_missing_file_errors(tmp_path):
    from skyward.cli.compute import _resolve_specifiers

    with pytest.raises(SystemExit):
        _resolve_specifiers(None, tmp_path / "nope.txt")


def test_install_broadcasts_by_default(stub, monkeypatch):
    monkeypatch.setattr("skyward.cli.compute._stream_run", AsyncMock(return_value=0))
    transport = stub(lambda req: httpx.Response(202, json={"id": "e1"}))
    with pytest.raises(SystemExit) as e:
        app(["install", "demo", "numpy"], exit_on_error=False)
    assert e.value.code == 0
    assert transport.calls[-1].url.params["mode"] == "broadcast"


def test_install_one_uses_run(stub, monkeypatch):
    monkeypatch.setattr("skyward.cli.compute._stream_run", AsyncMock(return_value=0))
    transport = stub(lambda req: httpx.Response(202, json={"id": "e1"}))
    with pytest.raises(SystemExit) as e:
        app(["install", "demo", "numpy", "--one"], exit_on_error=False)
    assert e.value.code == 0
    assert transport.calls[-1].url.params["mode"] == "run"


def test_install_409_not_ready(stub):
    stub(lambda req: httpx.Response(409, json={"error": "pool not ready", "status": "creating"}))
    with pytest.raises(SystemExit) as e:
        app(["install", "demo", "numpy"], exit_on_error=False)
    assert e.value.code == 1
