"""Tests for the log-export serializers and the sky log command."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from skyward.cli import app
from skyward.cli._log_export import to_ipynb, to_jsonl, to_markdown

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


def _ev(tid: str, node: int, msg: str, etype: str = "Log.Emitted") -> dict:
    return {"type": etype, "fields": {"task_id": tid, "node_id": node, "message": msg}}


# ── serializers ──────────────────────────────────────────────────


def test_jsonl_round_trips():
    events = [_ev("t1", 0, "a"), _ev("t1", 0, "b")]
    lines = [json.loads(line) for line in to_jsonl(events).strip().split("\n")]
    assert lines == events


def test_jsonl_empty():
    assert to_jsonl([]) == ""


def test_markdown_groups_by_task_and_node():
    md = to_markdown([_ev("t1", 0, "alpha"), _ev("t1", 1, "beta")])
    assert "## Task `t1`" in md
    assert "node 0" in md
    assert "node 1" in md
    assert "alpha" in md
    assert "beta" in md


def test_markdown_session_events_section():
    events = [{"type": "Pool.Provisioned", "fields": {"pool_name": "p"}}]
    md = to_markdown(events)
    assert "## Session events" in md
    assert "Pool.Provisioned" in md


def test_ipynb_is_nbformat4_with_source_and_output():
    events = [_ev("t1", 0, "output line")]
    nb = json.loads(to_ipynb(events, {"t1": "print('hi')"}))
    assert nb["nbformat"] == 4
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code
    assert "print('hi')" in "".join(code[0]["source"])
    out = code[0]["outputs"][0]
    assert out["output_type"] == "stream"
    assert "output line" in "".join(out["text"])


def test_ipynb_missing_source_is_empty_cell():
    nb = json.loads(to_ipynb([_ev("t1", 0, "x")], {}))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code[0]["source"] == []


# ── sky log command ──────────────────────────────────────────────


@pytest.fixture
def stub(monkeypatch):
    def install(response: httpx.Response):
        class _T(httpx.BaseTransport):
            def handle_request(self, request: httpx.Request) -> httpx.Response:
                return response

        monkeypatch.setattr(
            "skyward.cli.log.make_client",
            lambda url: httpx.Client(base_url=url or "http://localhost:7590", transport=_T()),
        )

    return install


def _log_response(**extra: Any) -> httpx.Response:
    body = {"name": "demo", "count": 1, "events": [_ev("t1", 0, "hello")], "sources": {"t1": "print(1)"}}
    body.update(extra)
    return httpx.Response(200, json=body)


def test_log_jsonl_stdout(stub, capsys):
    stub(_log_response())
    with pytest.raises(SystemExit) as e:
        app(["log", "demo"], exit_on_error=False)
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert json.loads(out.strip())["fields"]["message"] == "hello"


def test_log_ipynb_to_file(stub, tmp_path):
    stub(_log_response())
    out = tmp_path / "log.ipynb"
    with pytest.raises(SystemExit) as e:
        app(["log", "demo", "-o", str(out)], exit_on_error=False)
    assert e.value.code == 0
    nb = json.loads(out.read_text())
    assert nb["nbformat"] == 4


def test_log_404(stub):
    stub(httpx.Response(404, json={"error": "no history"}))
    with pytest.raises(SystemExit) as e:
        app(["log", "demo"], exit_on_error=False)
    assert e.value.code == 1
