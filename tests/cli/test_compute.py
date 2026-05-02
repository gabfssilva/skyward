"""Tests for sky compute commands (HTTP server CLI)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from skyward.cli import app

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class _StubTransport(httpx.BaseTransport):
    """Replays canned responses keyed by ``(method, path)``."""

    def __init__(self, routes: dict[tuple[str, str], httpx.Response]) -> None:
        self.routes = routes
        self.calls: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        key = (request.method, request.url.path)
        if key not in self.routes:
            return httpx.Response(404, json={"error": "no stub"})
        return self.routes[key]


@pytest.fixture
def stub_transport(monkeypatch):
    """Replace make_client to inject a stub transport per test."""

    captured: dict[str, Any] = {}

    def install(routes: dict[tuple[str, str], httpx.Response]) -> _StubTransport:
        transport = _StubTransport(routes)
        captured["transport"] = transport

        def fake_make_client(url: str | None) -> httpx.Client:
            base = url or "http://localhost:7590"
            return httpx.Client(base_url=base, transport=transport)

        monkeypatch.setattr("skyward.cli.compute.make_client", fake_make_client)
        monkeypatch.setattr("skyward.cli.server.make_client", fake_make_client)
        return transport

    return install


class TestComputeList:
    def test_lists_pools_as_json(self, stub_transport, capsys):
        stub_transport({
            ("GET", "/compute"): httpx.Response(
                200,
                json=[{"name": "demo", "current_nodes": 2, "concurrency": 4, "is_active": True}],
            ),
        })
        with pytest.raises(SystemExit, match="0"):
            app(["compute", "list", "--json"], exit_on_error=False)
        data = json.loads(capsys.readouterr().out)
        assert data[0]["name"] == "demo"

    def test_empty_list(self, stub_transport, capsys):
        stub_transport({("GET", "/compute"): httpx.Response(200, json=[])})
        with pytest.raises(SystemExit, match="0"):
            app(["compute", "list"], exit_on_error=False)
        assert "No pools" in capsys.readouterr().out


class TestComputeGet:
    def test_get_existing(self, stub_transport, capsys):
        stub_transport({
            ("GET", "/compute/demo"): httpx.Response(
                200,
                json={"name": "demo", "current_nodes": 1, "concurrency": 1, "is_active": True},
            ),
        })
        with pytest.raises(SystemExit, match="0"):
            app(["compute", "get", "demo", "--json"], exit_on_error=False)
        data = json.loads(capsys.readouterr().out)
        assert data["name"] == "demo"

    def test_get_404_exits_nonzero(self, stub_transport):
        stub_transport({("GET", "/compute/missing"): httpx.Response(404, json={"error": "not found"})})
        with pytest.raises(SystemExit) as exc:
            app(["compute", "get", "missing"], exit_on_error=False)
        assert exc.value.code == 1


class TestComputeDelete:
    def test_delete_ok(self, stub_transport, capsys):
        stub_transport({("DELETE", "/compute/demo"): httpx.Response(204)})
        with pytest.raises(SystemExit, match="0"):
            app(["compute", "delete", "demo"], exit_on_error=False)
        assert "deleted" in capsys.readouterr().out

    def test_delete_404(self, stub_transport):
        stub_transport({("DELETE", "/compute/missing"): httpx.Response(404, json={"error": "not found"})})
        with pytest.raises(SystemExit) as exc:
            app(["compute", "delete", "missing"], exit_on_error=False)
        assert exc.value.code == 1


class TestComputeCreate:
    def test_requires_pool_or_provider(self, capsys):
        with pytest.raises(SystemExit) as exc:
            app(["compute", "create"], exit_on_error=False)
        assert exc.value.code == 2

    def test_inline_provider_posts_payload(self, stub_transport):
        transport = stub_transport({
            ("POST", "/compute"): httpx.Response(
                202,
                json={
                    "name": "p1",
                    "status": "creating",
                    "current_nodes": 0,
                    "concurrency": 0,
                    "is_active": False,
                    "error": None,
                },
            ),
        })
        with pytest.raises(SystemExit, match="0"):
            app(
                ["compute", "create", "--provider", "vastai", "--name", "p1"],
                exit_on_error=False,
            )
        post = next(c for c in transport.calls if c.method == "POST")
        assert post.url.params["name"] == "p1"
        assert post.headers["content-type"] == "application/octet-stream"

        from skyward.api.spec import Options, Spec
        from skyward.server.wire import decode

        specs, options = decode(post.content)
        assert isinstance(options, Options)
        assert isinstance(specs[0], Spec)

    def test_unknown_provider(self, capsys):
        with pytest.raises(SystemExit) as exc:
            app(["compute", "create", "--provider", "made-up"], exit_on_error=False)
        assert exc.value.code == 2
        assert "Unknown provider" in capsys.readouterr().out


class TestUrlResolution:
    def test_flag_beats_env(self, monkeypatch):
        from skyward.cli._client import resolve_server_url

        monkeypatch.setenv("SKYWARD_SERVER_URL", "http://from-env:1234")
        assert resolve_server_url("http://from-flag:5678") == "http://from-flag:5678"

    def test_env_used_when_no_flag(self, monkeypatch):
        from skyward.cli._client import resolve_server_url

        monkeypatch.setenv("SKYWARD_SERVER_URL", "http://from-env:1234/")
        assert resolve_server_url(None) == "http://from-env:1234"

    def test_default_when_neither(self, monkeypatch):
        from skyward.cli._client import resolve_server_url

        monkeypatch.delenv("SKYWARD_SERVER_URL", raising=False)
        assert resolve_server_url(None) == "http://localhost:7590"
