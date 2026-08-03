import pytest

pytest.importorskip("cyclopts", reason="the sky CLI needs: pip install 'skyward[cli]'")

from skyward.cli import app
from skyward.cli.monitor import monitor


@pytest.fixture
def database(tmp_path, monkeypatch):
    monkeypatch.delenv("SKYWARD_URL", raising=False)
    return tmp_path / "cli.sqlite"


@pytest.mark.unit
def test_monitoring_an_absent_compute_reports_a_refusal(database):
    with pytest.raises(SystemExit, match="not_found"):
        monitor("absent", database=database)


@pytest.mark.unit
def test_an_interrupt_leaves_without_a_traceback(monkeypatch):
    def interrupted(work, *, url=None, database=None):
        raise KeyboardInterrupt

    monkeypatch.setattr("skyward.cli.monitor.call", interrupted)

    assert monitor("pool-1") is None


@pytest.mark.unit
def test_monitor_is_registered_at_the_top_level():
    assert "monitor" in {name for command in app.subapps for name in command.name}


@pytest.mark.unit
def test_monitor_selects_the_console_mode(monkeypatch):
    selected = None

    class Compute:
        id = "cmp_1"

    class Client:
        async def call(self, *args):
            return Compute()

    class Follower:
        async def follow(self):
            return None

    def watching(client, compute, out=None, *, mode):
        nonlocal selected
        selected = mode
        return Follower()

    def called(work, *, url=None, database=None):
        import asyncio

        return asyncio.run(work(Client()))

    monkeypatch.setattr("skyward.cli.monitor.watcher", watching)
    monkeypatch.setattr("skyward.cli.monitor.call", called)

    monitor("pool-1", mode="log")

    assert selected == "log"
