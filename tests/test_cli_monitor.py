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
