import io
import sys
from contextlib import redirect_stdout

import pytest

pytest.importorskip("cyclopts", reason="the sky CLI needs: pip install 'skyward[cli]'")

from skyward._version import __version__
from skyward.cli import app, compute_app, config_app, log_app, offers_app, providers_app, server_app, version
from skyward.cli._client import resolve
from skyward.cli._output import render


def captured(call):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        call()
    return buffer.getvalue()


@pytest.mark.unit
def test_app_is_named_sky():
    assert tuple(app.name) == ("sky",)


@pytest.mark.unit
@pytest.mark.parametrize(
    "sub",
    [offers_app, providers_app, config_app, server_app, compute_app, log_app],
)
def test_sub_apps_are_registered(sub):
    assert sub in list(app.subapps)


@pytest.mark.unit
def test_version_prints_skyward_and_python():
    assert captured(version) == f"skyward {__version__}, python {sys.version.split()[0]}\n"


@pytest.mark.unit
def test_render_table_aligns_columns():
    output = captured(lambda: render(["id", "state"], [["a", "running"], ["longer-id", None]]))
    assert output == "id         state\na          running\nlonger-id  -\n"


@pytest.mark.unit
def test_render_table_without_rows_prints_header():
    assert captured(lambda: render(["id", "state"], [])) == "id  state\n"


@pytest.mark.unit
def test_render_json_emits_objects():
    output = captured(lambda: render(["id", "nodes"], [["a", 2]], output="json"))
    assert output == '[\n  {\n    "id": "a",\n    "nodes": "2"\n  }\n]\n'


@pytest.mark.unit
def test_resolve_prefers_the_flag_over_the_env(monkeypatch):
    monkeypatch.setenv("SKYWARD_URL", "http://env:7590")
    assert resolve("http://flag:7590/") == "http://flag:7590"


@pytest.mark.unit
def test_resolve_falls_back_to_the_env(monkeypatch):
    monkeypatch.setenv("SKYWARD_URL", "http://env:7590/")
    assert resolve(None) == "http://env:7590"


@pytest.mark.unit
def test_resolve_is_none_without_a_daemon(monkeypatch):
    monkeypatch.delenv("SKYWARD_URL", raising=False)
    assert resolve(None) is None
