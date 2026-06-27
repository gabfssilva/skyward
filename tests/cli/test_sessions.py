"""Tests for the top-level session verbs (new / sessions / status / stop)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from skyward.cli import _session_store as ss
from skyward.cli import app

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class _StubTransport(httpx.BaseTransport):
    def __init__(self, routes: dict[tuple[str, str], httpx.Response]) -> None:
        self.routes = routes
        self.calls: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        return self.routes.get(
            (request.method, request.url.path),
            httpx.Response(404, json={"error": "no stub"}),
        )


class _ConnErrTransport(httpx.BaseTransport):
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")


@pytest.fixture
def session_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "SESSION_FILE", tmp_path / "current-session")


@pytest.fixture
def stub(monkeypatch):
    captured: dict[str, Any] = {}

    def install(routes: dict[tuple[str, str], httpx.Response] | None) -> None:
        transport: httpx.BaseTransport = (
            _ConnErrTransport() if routes is None else _StubTransport(routes)
        )
        captured["transport"] = transport

        def fake_make_client(url: str | None) -> httpx.Client:
            return httpx.Client(base_url=url or "http://localhost:7590", transport=transport)

        for mod in ("sessions", "compute", "_session_store"):
            monkeypatch.setattr(f"skyward.cli.{mod}.make_client", fake_make_client)

    return install


def _pool(name: str, status: str = "ready") -> dict:
    return {
        "name": name, "status": status, "current_nodes": 1,
        "concurrency": 1, "is_active": status == "ready",
    }


def test_new_persists_current(stub, session_file):
    stub({("POST", "/compute"): httpx.Response(202, json=_pool("s1", "creating"))})
    with pytest.raises(SystemExit) as e:
        app(["new", "s1", "--provider", "container"], exit_on_error=False)
    assert e.value.code == 0
    assert ss.read_current_session() == "s1"


def test_sessions_marks_current(stub, session_file, capsys):
    ss.write_current_session("s1")
    stub({("GET", "/compute"): httpx.Response(200, json=[_pool("s1"), _pool("s2")])})
    with pytest.raises(SystemExit) as e:
        app(["sessions", "--json"], exit_on_error=False)
    assert e.value.code == 0
    data = json.loads(capsys.readouterr().out)
    by_name = {p["name"]: p["current"] for p in data}
    assert by_name == {"s1": True, "s2": False}


def test_status_nodes_unavailable_when_not_ready(stub, session_file, capsys):
    stub({
        ("GET", "/compute/x"): httpx.Response(200, json=_pool("x", "creating")),
        ("GET", "/compute/x/nodes"): httpx.Response(
            409, json={"error": "pool not ready", "status": "creating"},
        ),
    })
    with pytest.raises(SystemExit) as e:
        app(["status", "-s", "x", "--json"], exit_on_error=False)
    assert e.value.code == 0
    assert json.loads(capsys.readouterr().out)["nodes"] == []


def test_status_includes_node_table(stub, session_file, capsys):
    stub({
        ("GET", "/compute/x"): httpx.Response(200, json=_pool("x")),
        ("GET", "/compute/x/nodes"): httpx.Response(200, json={
            "name": "x",
            "nodes": [
                {"rank": 0, "is_head": True, "status": "ready", "instance_id": "i-0",
                 "ip": "1.1.1.1", "ssh_user": "ubuntu", "ssh_port": 22},
            ],
        }),
    })
    with pytest.raises(SystemExit) as e:
        app(["status", "-s", "x", "--json"], exit_on_error=False)
    assert e.value.code == 0
    nodes = json.loads(capsys.readouterr().out)["nodes"]
    assert nodes[0]["is_head"] is True


def test_stop_clears_current_on_auto_resolve(stub, session_file):
    ss.write_current_session("s1")
    stub({
        ("GET", "/compute"): httpx.Response(200, json=[_pool("s1")]),
        ("DELETE", "/compute/s1"): httpx.Response(204),
    })
    with pytest.raises(SystemExit) as e:
        app(["stop"], exit_on_error=False)
    assert e.value.code == 0
    assert ss.read_current_session() is None


def test_stop_404_exits_nonzero(stub, session_file):
    stub({("DELETE", "/compute/x"): httpx.Response(404, json={"error": "not found"})})
    with pytest.raises(SystemExit) as e:
        app(["stop", "-s", "x"], exit_on_error=False)
    assert e.value.code == 1


def test_sessions_connect_error_exits(stub, session_file):
    stub(None)
    with pytest.raises(SystemExit) as e:
        app(["sessions"], exit_on_error=False)
    assert e.value.code == 1
