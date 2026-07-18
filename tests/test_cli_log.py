import asyncio
import io
import json
from contextlib import redirect_stdout

import pytest

pytest.importorskip("cyclopts", reason="the sky CLI needs: pip install 'skyward[cli]'")

from skyward.cli import log_app
from skyward.cli.log import _markdown, _read, _serialize, export_log, show_log


class FakeClient:
    def __init__(self, events, hang=False):
        self._events = events
        self._hang = hang

    async def events(self, compute):
        for name, payload in self._events:
            yield name, json.dumps(payload).encode()
        if self._hang:
            await asyncio.Event().wait()


EVENTS = [
    ("node.console", {"compute": "c1", "node": "n0", "content": "hello"}),
    ("node.console", {"compute": "c1", "node": "n1", "content": "world"}),
    ("task.failed", {"compute": "c1", "task": "t1", "error": "boom"}),
]


def captured(call):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        call()
    return buffer.getvalue()


def patched(monkeypatch, events=EVENTS, hang=False):
    import skyward.cli.log as module

    def call(work, *, url=None, database=None):
        return asyncio.run(work(FakeClient(events, hang)))

    monkeypatch.setattr(module, "call", call)


@pytest.mark.unit
def test_export_is_registered():
    assert "export" in log_app


@pytest.mark.unit
def test_read_stops_when_the_replay_goes_quiet():
    events = asyncio.run(_read(FakeClient(EVENTS, hang=True), "c1", 0.1, None))
    assert [name for name, _ in events] == ["node.console", "node.console", "task.failed"]


@pytest.mark.unit
def test_show_prints_a_table(monkeypatch):
    patched(monkeypatch)
    output = captured(lambda: show_log("c1", idle=0.1))
    assert output.splitlines()[0].split() == ["event", "who", "detail"]
    assert "hello" in output and "boom" in output


@pytest.mark.unit
def test_show_limit_keeps_the_last_events(monkeypatch):
    patched(monkeypatch)
    body = json.loads(captured(lambda: show_log("c1", idle=0.1, limit=1, output="json")))
    assert body == [{"event": "task.failed", "compute": "c1", "task": "t1", "error": "boom"}]


@pytest.mark.unit
def test_follow_echoes_each_event_as_it_arrives(monkeypatch):
    patched(monkeypatch, events=EVENTS[:1])
    line = captured(lambda: show_log("c1", follow=True, output="json")).strip()
    assert json.loads(line) == {"event": "node.console", "compute": "c1", "node": "n0", "content": "hello"}


@pytest.mark.unit
def test_export_writes_jsonl(monkeypatch, tmp_path):
    patched(monkeypatch)
    target = tmp_path / "log.jsonl"
    captured(lambda: export_log("c1", target, idle=0.1))
    assert [json.loads(line)["event"] for line in target.read_text().splitlines()] == [name for name, _ in EVENTS]


@pytest.mark.unit
def test_export_writes_markdown(monkeypatch, tmp_path):
    patched(monkeypatch)
    target = tmp_path / "log.md"
    captured(lambda: export_log("c1", target, idle=0.1))
    text = target.read_text()
    assert "## node `n0`" in text and "hello" in text and "**task.failed**" in text


@pytest.mark.unit
def test_export_rejects_an_unknown_suffix():
    with pytest.raises(SystemExit):
        _serialize([], ".pdf")


@pytest.mark.unit
def test_markdown_groups_console_by_node():
    text = _markdown(EVENTS)
    assert text.index("## node `n0`") < text.index("## node `n1`") < text.index("## Events")
