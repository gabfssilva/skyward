"""Tests for the sky compute file commands (ls / rm / upload / download)."""

from __future__ import annotations

import json
from typing import Any

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

    def install(responder) -> _StubTransport:
        transport = _StubTransport(responder)
        captured["transport"] = transport
        monkeypatch.setattr(
            "skyward.cli.compute.make_client",
            lambda url: httpx.Client(base_url=url or "http://localhost:7590", transport=transport),
        )
        return transport

    return install


def test_ls_prints_listing(stub, capsys):
    stub(lambda req: httpx.Response(200, json={"results": [
        {"node_id": 0, "success": True, "listing": "total 0\n/opt", "error": None},
    ]}))
    with pytest.raises(SystemExit) as e:
        app(["compute", "ls", "demo", "/opt"], exit_on_error=False)
    assert e.value.code == 0
    assert "/opt" in capsys.readouterr().out


def test_rm_404(stub):
    stub(lambda req: httpx.Response(404, json={"error": "not found"}))
    with pytest.raises(SystemExit) as e:
        app(["compute", "rm", "demo", "/x"], exit_on_error=False)
    assert e.value.code == 1


def test_ls_409_not_ready(stub):
    stub(lambda req: httpx.Response(409, json={"error": "pool not ready", "status": "creating"}))
    with pytest.raises(SystemExit) as e:
        app(["compute", "ls", "demo", "/x"], exit_on_error=False)
    assert e.value.code == 1


def test_upload_sends_bytes(stub, tmp_path):
    local = tmp_path / "a.bin"
    local.write_bytes(b"\x00\x01payload")
    transport = stub(lambda req: httpx.Response(200, json={"results": [
        {"node_id": 0, "success": True, "listing": "", "error": None},
    ]}))
    with pytest.raises(SystemExit) as e:
        app(["compute", "upload", "demo", str(local), "/tmp/a.bin"], exit_on_error=False)
    assert e.value.code == 0
    put = transport.calls[-1]
    assert put.method == "PUT"
    assert put.content == b"\x00\x01payload"
    assert put.url.params["path"] == "/tmp/a.bin"


def test_download_writes_file(stub, tmp_path):
    stub(lambda req: httpx.Response(200, content=b"downloaded-bytes"))
    dest = tmp_path / "out.bin"
    with pytest.raises(SystemExit) as e:
        app(["compute", "download", "demo", "/tmp/r.bin", str(dest)], exit_on_error=False)
    assert e.value.code == 0
    assert dest.read_bytes() == b"downloaded-bytes"


def test_upload_missing_local_file(stub, tmp_path):
    stub(lambda req: httpx.Response(200, json={"results": []}))
    with pytest.raises(SystemExit) as e:
        app(["compute", "upload", "demo", str(tmp_path / "nope.bin"), "/tmp/x"], exit_on_error=False)
    assert e.value.code == 2


def test_ls_json_output(stub, capsys):
    stub(lambda req: httpx.Response(200, json={"results": [
        {"node_id": 0, "success": True, "listing": "x", "error": None},
    ]}))
    with pytest.raises(SystemExit) as e:
        app(["compute", "ls", "demo", "/opt", "--json"], exit_on_error=False)
    assert e.value.code == 0
    assert json.loads(capsys.readouterr().out)[0]["node_id"] == 0
