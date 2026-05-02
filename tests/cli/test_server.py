"""Tests for sky server commands."""

from __future__ import annotations

import json

import httpx
import pytest

from skyward.cli import app

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class _StubTransport(httpx.BaseTransport):
    def __init__(self, response: httpx.Response | type[BaseException]) -> None:
        self.response = response
        self.calls: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if isinstance(self.response, type) and issubclass(self.response, BaseException):
            raise self.response("simulated", request=request)
        return self.response


@pytest.fixture
def stub_transport(monkeypatch):
    def install(response: httpx.Response | type[BaseException]) -> _StubTransport:
        transport = _StubTransport(response)

        def fake_make_client(url: str | None) -> httpx.Client:
            return httpx.Client(base_url=url or "http://localhost:7590", transport=transport)

        monkeypatch.setattr("skyward.cli.server.make_client", fake_make_client)
        return transport

    return install


class TestStatus:
    def test_renders_health_payload(self, stub_transport, capsys):
        stub_transport(httpx.Response(
            200,
            json={"status": "ok", "version": "0.1.0", "pools": 2, "executions": 5},
        ))
        with pytest.raises(SystemExit, match="0"):
            app(["server", "status"], exit_on_error=False)
        out = capsys.readouterr().out
        assert "0.1.0" in out
        assert "2" in out

    def test_json_output(self, stub_transport, capsys):
        stub_transport(httpx.Response(
            200,
            json={"status": "ok", "version": "0.1.0", "pools": 0, "executions": 0},
        ))
        with pytest.raises(SystemExit, match="0"):
            app(["server", "status", "--json"], exit_on_error=False)
        data = json.loads(capsys.readouterr().out)
        assert data["status"] == "ok"
        assert data["url"].startswith("http://")

    def test_unreachable_exits_nonzero(self, stub_transport):
        stub_transport(httpx.ConnectError)
        with pytest.raises(SystemExit) as exc:
            app(["server", "status"], exit_on_error=False)
        assert exc.value.code == 1


@pytest.fixture
def isolated_runtime(monkeypatch, tmp_path):
    """Redirect the PID/log files to a per-test tmp dir."""
    pid_file = tmp_path / "server.pid"
    log_file = tmp_path / "server.log"
    monkeypatch.setattr("skyward.cli.server._RUNTIME_DIR", tmp_path)
    monkeypatch.setattr("skyward.cli.server._PID_FILE", pid_file)
    monkeypatch.setattr("skyward.cli.server._LOG_FILE", log_file)
    return pid_file


class TestStop:
    def test_sends_post_shutdown(self, stub_transport, capsys, isolated_runtime, monkeypatch):
        transport = stub_transport(httpx.Response(202))
        # No PID file — _wait_for_exit short-circuits via _read_pid() == None
        with pytest.raises(SystemExit, match="0"):
            app(["server", "stop"], exit_on_error=False)
        assert transport.calls[0].method == "POST"
        assert transport.calls[0].url.path == "/shutdown"
        assert "Shutdown requested" in capsys.readouterr().out

    def test_clears_pid_file_after_daemon_exits(
        self, stub_transport, isolated_runtime, monkeypatch,
    ):
        from skyward.cli import server as server_mod

        isolated_runtime.write_text("12345")
        stub_transport(httpx.Response(202))
        monkeypatch.setattr(server_mod, "_is_alive", lambda pid: False)

        with pytest.raises(SystemExit, match="0"):
            app(["server", "stop"], exit_on_error=False)
        assert not isolated_runtime.exists()

    def test_unreachable_with_stale_pid_cleans_up(self, stub_transport, isolated_runtime, monkeypatch, capsys):
        from skyward.cli import server as server_mod

        isolated_runtime.write_text("99999")
        monkeypatch.setattr(server_mod, "_is_alive", lambda pid: False)
        stub_transport(httpx.ConnectError)

        with pytest.raises(SystemExit, match="0"):
            app(["server", "stop"], exit_on_error=False)
        assert not isolated_runtime.exists()
        assert "stale pid" in capsys.readouterr().out.lower()

    def test_unreachable_no_pid(self, stub_transport, isolated_runtime):
        stub_transport(httpx.ConnectError)
        with pytest.raises(SystemExit) as exc:
            app(["server", "stop"], exit_on_error=False)
        assert exc.value.code == 1


class TestStartForeground:
    def test_uvicorn_invoked_with_expected_config(self, monkeypatch):
        captured: dict = {}

        class FakeServer:
            def __init__(self, config):  # noqa: ANN001
                captured["config"] = config

            def run(self):  # noqa: ANN001
                captured["ran"] = True

        class FakeUvicorn:
            Config = lambda *a, **kw: ("config", a, kw)  # noqa: E731
            Server = FakeServer

        monkeypatch.setitem(__import__("sys").modules, "uvicorn", FakeUvicorn)

        with pytest.raises(SystemExit, match="0"):
            app(
                ["server", "start", "--foreground", "--host", "0.0.0.0", "--port", "9999"],
                exit_on_error=False,
            )

        assert captured["ran"] is True
        _, args, kwargs = captured["config"]
        assert args[0] == "skyward.server:app"
        assert kwargs["host"] == "0.0.0.0"
        assert kwargs["port"] == 9999
        assert kwargs["reload"] is False

    def test_reload_implies_foreground(self, monkeypatch):
        captured: dict = {}

        class FakeServer:
            def __init__(self, config):  # noqa: ANN001
                captured["config"] = config

            def run(self):  # noqa: ANN001
                captured["ran"] = True

        class FakeUvicorn:
            Config = lambda *a, **kw: ("config", a, kw)  # noqa: E731
            Server = FakeServer

        monkeypatch.setitem(__import__("sys").modules, "uvicorn", FakeUvicorn)

        with pytest.raises(SystemExit, match="0"):
            app(["server", "start", "--reload"], exit_on_error=False)
        # No daemon spawn even though --foreground wasn't passed
        assert captured["ran"] is True


class TestStartDaemon:
    def test_spawns_subprocess_writes_pid_polls_health(
        self, isolated_runtime, monkeypatch, capsys,
    ):
        from skyward.cli import server as server_mod

        spawned: dict = {}

        class FakeProc:
            pid = 4242

        def fake_spawn(host: str, port: int) -> FakeProc:
            spawned["host"] = host
            spawned["port"] = port
            return FakeProc()

        monkeypatch.setattr(server_mod, "_spawn_daemon", fake_spawn)
        monkeypatch.setattr(server_mod, "_wait_for_health", lambda url, *, timeout: True)

        with pytest.raises(SystemExit, match="0"):
            app(["server", "start", "--port", "9999"], exit_on_error=False)

        assert spawned == {"host": "127.0.0.1", "port": 9999}
        assert isolated_runtime.read_text() == "4242"
        out = capsys.readouterr().out
        assert "Server running" in out
        assert "9999" in out

    def test_aborts_when_health_fails(
        self, isolated_runtime, monkeypatch, capsys,
    ):
        from skyward.cli import server as server_mod

        killed: list[int] = []

        class FakeProc:
            pid = 7777

        monkeypatch.setattr(server_mod, "_spawn_daemon", lambda h, p: FakeProc())
        monkeypatch.setattr(server_mod, "_wait_for_health", lambda url, *, timeout: False)
        monkeypatch.setattr(
            server_mod.os, "kill", lambda pid, sig: killed.append(pid),
        )

        with pytest.raises(SystemExit) as exc:
            app(["server", "start", "--timeout", "1"], exit_on_error=False)
        assert exc.value.code == 1
        assert killed == [7777]
        assert not isolated_runtime.exists()
        assert "did not become healthy" in capsys.readouterr().out

    def test_refuses_when_already_running(self, isolated_runtime, monkeypatch, capsys):
        from skyward.cli import server as server_mod

        isolated_runtime.write_text("11111")
        monkeypatch.setattr(server_mod, "_is_alive", lambda pid: True)

        with pytest.raises(SystemExit) as exc:
            app(["server", "start"], exit_on_error=False)
        assert exc.value.code == 1
        assert "already running" in capsys.readouterr().out

    def test_clears_stale_pid_then_starts(self, isolated_runtime, monkeypatch):
        from skyward.cli import server as server_mod

        isolated_runtime.write_text("33333")
        monkeypatch.setattr(server_mod, "_is_alive", lambda pid: False)

        class FakeProc:
            pid = 4242

        monkeypatch.setattr(server_mod, "_spawn_daemon", lambda h, p: FakeProc())
        monkeypatch.setattr(server_mod, "_wait_for_health", lambda url, *, timeout: True)

        with pytest.raises(SystemExit, match="0"):
            app(["server", "start"], exit_on_error=False)
        assert isolated_runtime.read_text() == "4242"
