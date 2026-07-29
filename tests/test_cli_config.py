import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

pytest.importorskip("cyclopts", reason="the sky CLI needs: pip install 'skyward[cli]'")

from skyward.cli.config import config_path, config_show, config_validate


def captured(call):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        call()
    return buffer.getvalue()


def settings(output):
    return {row["setting"]: row["value"] for row in json.loads(output)}


@pytest.mark.unit
def test_path_reports_the_default_database(monkeypatch):
    from skyward.server.persistence.db import DEFAULT_PATH

    monkeypatch.delenv("SKYWARD_URL", raising=False)
    rows = settings(captured(lambda: config_path(output="json")))

    assert rows["database"] == str(DEFAULT_PATH)
    assert rows["url"] == "-"


@pytest.mark.unit
def test_path_reports_the_resolved_url(monkeypatch):
    monkeypatch.setenv("SKYWARD_URL", "http://env:7590/")
    assert settings(captured(lambda: config_path(output="json")))["url"] == "http://env:7590"


@pytest.mark.unit
def test_show_names_the_source_of_the_url(monkeypatch):
    monkeypatch.setenv("SKYWARD_URL", "http://env:7590")

    assert settings(captured(lambda: config_show(output="json")))["source"] == "environment"
    assert settings(captured(lambda: config_show(url="http://flag:7590", output="json")))["source"] == "flag"

    monkeypatch.delenv("SKYWARD_URL", raising=False)
    assert settings(captured(lambda: config_show(output="json")))["source"] == "embedded"


@pytest.mark.unit
def test_show_reports_whether_the_database_exists(tmp_path, monkeypatch):
    monkeypatch.delenv("SKYWARD_URL", raising=False)
    missing = tmp_path / "absent.sqlite"

    rows = settings(captured(lambda: config_show(database=missing, output="json")))
    assert rows["database"] == str(missing)
    assert rows["database exists"] == "false"

    missing.touch()
    assert settings(captured(lambda: config_show(database=missing, output="json")))["database exists"] == "true"


@pytest.mark.unit
def test_validate_passes_against_an_embedded_daemon(tmp_path, monkeypatch):
    monkeypatch.delenv("SKYWARD_URL", raising=False)
    output = captured(lambda: config_validate(database=tmp_path / "skyward.sqlite", output="json"))

    checks = {row["check"]: row["status"] for row in json.loads(output)}
    assert checks["daemon (embedded)"] == "ok"


@pytest.mark.unit
def test_validate_exits_nonzero_when_the_daemon_is_unreachable(monkeypatch):
    monkeypatch.setenv("SKYWARD_URL", "http://127.0.0.1:1/")

    with pytest.raises(SystemExit) as exit, redirect_stdout(io.StringIO()):
        config_validate(output="json")

    assert exit.value.code == 1


@pytest.mark.unit
def test_path_accepts_an_explicit_database(monkeypatch):
    monkeypatch.delenv("SKYWARD_URL", raising=False)
    rows = settings(captured(lambda: config_path(database=Path("/tmp/other.sqlite"), output="json")))

    assert rows["database"] == "/tmp/other.sqlite"
