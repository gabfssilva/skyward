import io
import json
from contextlib import redirect_stdout

import pytest

pytest.importorskip("cyclopts", reason="the sky CLI needs: pip install 'skyward[cli]'")

from skyward.cli import app
from skyward.cli.sessions import sessions, status, stop


def captured(call):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        call()
    return buffer.getvalue()


@pytest.fixture
def database(tmp_path, monkeypatch):
    monkeypatch.delenv("SKYWARD_URL", raising=False)
    return tmp_path / "cli.sqlite"


@pytest.mark.unit
def test_status_without_a_ref_lists_every_compute(database):
    assert json.loads(captured(lambda: status(database=database, output="json"))) == []


@pytest.mark.unit
def test_sessions_lists_every_compute(database):
    assert json.loads(captured(lambda: sessions(database=database, output="json"))) == []


@pytest.mark.unit
def test_status_with_a_ref_reads_that_one_and_reports_a_refusal(database):
    with pytest.raises(SystemExit, match="not_found"):
        status("absent", database=database)


@pytest.mark.unit
def test_stop_reports_a_refusal_instead_of_raising(database):
    with pytest.raises(SystemExit, match="not_found"):
        stop("absent", database=database)


@pytest.mark.unit
def test_status_routes_by_whether_it_was_given_a_ref(monkeypatch):
    seen: list[tuple[str, object]] = []
    monkeypatch.setattr("skyward.cli.sessions.list_computes", lambda **kwargs: seen.append(("list", None)))
    monkeypatch.setattr("skyward.cli.sessions.get_compute", lambda ref, **kwargs: seen.append(("get", ref)))

    status()
    status("pool-1")

    assert seen == [("list", None), ("get", "pool-1")]


@pytest.mark.unit
def test_stop_deletes_through_the_compute_command(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr("skyward.cli.sessions.delete_compute", lambda ref, **kwargs: seen.append(ref))

    stop("pool-1")

    assert seen == ["pool-1"]


@pytest.mark.unit
def test_the_verbs_are_registered_at_the_top_level():
    assert {"status", "sessions", "stop"} <= {name for command in app.subapps for name in command.name}
