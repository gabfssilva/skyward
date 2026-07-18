import io
import json
from contextlib import redirect_stdout

import pytest

pytest.importorskip("cyclopts", reason="the sky CLI needs: pip install 'skyward[cli]'")

from skyward.cli.compute import FACTORIES, create_compute, delete_compute, get_compute, list_computes, view_compute


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
def test_list_on_an_empty_database_prints_the_header(database):
    assert json.loads(captured(lambda: list_computes(database=database, output="json"))) == []


@pytest.mark.unit
def test_get_reports_a_refusal_instead_of_raising(database):
    with pytest.raises(SystemExit, match="not_found"):
        get_compute("absent", database=database)


@pytest.mark.unit
def test_delete_reports_a_refusal_instead_of_raising(database):
    with pytest.raises(SystemExit, match="not_found"):
        delete_compute("absent", database=database)


@pytest.mark.unit
def test_view_reports_a_refusal_instead_of_raising(database):
    with pytest.raises(SystemExit, match="not_found"):
        view_compute("absent", database=database)


@pytest.mark.unit
def test_create_refuses_an_unknown_provider(database):
    with pytest.raises(SystemExit, match="unknown provider"):
        create_compute(provider="nowhere", database=database)


@pytest.mark.unit
def test_every_factory_answers_with_its_own_kind():
    assert {kind: build().kind for kind, build in FACTORIES.items()} == {kind: kind for kind in FACTORIES}
