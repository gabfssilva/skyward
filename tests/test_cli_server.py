import io
import json
import os
import signal
from contextlib import redirect_stdout

import pytest

pytest.importorskip("cyclopts", reason="the sky CLI needs: pip install 'skyward[cli]'")

from skyward.cli import server as cli


@pytest.fixture
def pidfile(tmp_path, monkeypatch):
    path = tmp_path / "server.pid"
    monkeypatch.setattr(cli, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(cli, "PID_FILE", path)
    monkeypatch.setattr(cli, "LOG_FILE", tmp_path / "server.log")
    return path


def captured(call):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        call()
    return buffer.getvalue()


@pytest.mark.unit
def test_endpoint_prefers_the_resolved_url(monkeypatch):
    monkeypatch.setenv("SKYWARD_URL", "http://env:9000/")
    assert cli.endpoint(None, "127.0.0.1", 7590) == "http://env:9000"


@pytest.mark.unit
def test_endpoint_falls_back_to_the_bind_address(monkeypatch):
    monkeypatch.delenv("SKYWARD_URL", raising=False)
    assert cli.endpoint(None, "0.0.0.0", 8080) == "http://0.0.0.0:8080"


@pytest.mark.unit
def test_pid_is_none_without_a_pidfile(pidfile):
    assert cli.pid() is None


@pytest.mark.unit
def test_pid_is_none_when_the_pidfile_is_garbage(pidfile):
    pidfile.write_text("not a pid")
    assert cli.pid() is None


@pytest.mark.unit
def test_pid_reads_the_recorded_process(pidfile):
    pidfile.write_text(" 4242\n")
    assert cli.pid() == 4242


@pytest.mark.unit
def test_alive_sees_this_process():
    assert cli.alive(os.getpid())


@pytest.mark.unit
def test_alive_is_false_for_a_dead_process(monkeypatch):
    monkeypatch.setattr(cli.os, "kill", lambda *_: (_ for _ in ()).throw(ProcessLookupError()))
    assert not cli.alive(4242)


@pytest.mark.unit
def test_start_refuses_when_one_is_already_running(pidfile, monkeypatch):
    pidfile.write_text(str(os.getpid()))
    monkeypatch.setattr(cli, "_require_uvicorn", lambda: None)
    with pytest.raises(SystemExit, match="already running"):
        cli.start()


@pytest.mark.unit
def test_start_reports_the_endpoint_and_pid(pidfile, monkeypatch):
    monkeypatch.setattr(cli, "_require_uvicorn", lambda: None)
    monkeypatch.setattr(cli, "_spawn", lambda host, port: 4242)
    monkeypatch.setattr(cli, "_wait_live", lambda target, timeout: True)
    assert "http://127.0.0.1:7590 (pid 4242)" in captured(cli.start)


@pytest.mark.unit
def test_start_clears_the_pidfile_when_nothing_answers(pidfile, monkeypatch):
    pidfile.write_text("4242")
    monkeypatch.setattr(cli, "_require_uvicorn", lambda: None)
    monkeypatch.setattr(cli, "alive", lambda process: False)
    monkeypatch.setattr(cli, "_spawn", lambda host, port: pidfile.write_text("99") or 99)
    monkeypatch.setattr(cli, "_wait_live", lambda target, timeout: False)
    with pytest.raises(SystemExit, match="no answer"):
        cli.start()
    assert not pidfile.exists()


@pytest.mark.unit
def test_stop_without_a_pidfile_exits(pidfile):
    with pytest.raises(SystemExit, match="nothing to stop"):
        cli.stop()


@pytest.mark.unit
def test_stop_clears_a_stale_pidfile(pidfile, monkeypatch):
    pidfile.write_text("4242")
    monkeypatch.setattr(cli, "alive", lambda process: False)
    assert "stale pid 4242" in captured(cli.stop)
    assert not pidfile.exists()


@pytest.mark.unit
def test_stop_signals_the_recorded_process(pidfile, monkeypatch):
    pidfile.write_text("4242")
    signalled: list[tuple[int, int]] = []
    monkeypatch.setattr(cli, "alive", lambda process: not signalled)
    monkeypatch.setattr(cli.os, "kill", lambda process, sig: signalled.append((process, sig)))
    assert "stopped (pid 4242)" in captured(cli.stop)
    assert signalled == [(4242, signal.SIGTERM)]
    assert not pidfile.exists()


@pytest.mark.unit
def test_stop_reports_a_process_that_stayed(pidfile, monkeypatch):
    pidfile.write_text("4242")
    monkeypatch.setattr(cli, "alive", lambda process: True)
    monkeypatch.setattr(cli.os, "kill", lambda process, sig: None)
    monkeypatch.setattr(cli, "_wait_exit", lambda process, timeout: False)
    with pytest.raises(SystemExit, match="still alive"):
        cli.stop()
    assert pidfile.exists()


@pytest.mark.unit
def test_status_reports_the_probe_and_the_pid(pidfile, monkeypatch):
    pidfile.write_text("4242")
    monkeypatch.setattr(cli, "alive", lambda process: True)
    monkeypatch.setattr(cli, "live", lambda target: True)
    rows = json.loads(captured(lambda: cli.status(url="http://host:7590", output="json")))
    assert rows == [{"url": "http://host:7590", "pid": "4242", "live": "True"}]


@pytest.mark.unit
def test_status_without_a_daemon_shows_no_pid(pidfile, monkeypatch):
    monkeypatch.delenv("SKYWARD_URL", raising=False)
    monkeypatch.setattr(cli, "live", lambda target: False)
    rows = json.loads(captured(lambda: cli.status(output="json")))
    assert rows == [{"url": "http://127.0.0.1:7590", "pid": "-", "live": "False"}]


@pytest.mark.unit
async def test_probe_is_false_when_nothing_answers():
    from skyward.core.client import Client

    client = await Client.remote("http://127.0.0.1:1")
    try:
        assert not await cli.probe(client)
    finally:
        await client.close()
