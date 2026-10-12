"""Where a subprocess's print ends up.

Thread-mode output goes through the worker's redirected stdout into the journal
the daemon tails. A process executor's child inherits a raw fd instead — the
worker's log file, which nobody streams — so the child has to redirect for
itself, and say which task was speaking.
"""

import contextvars
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path
from types import MappingProxyType

import pytest

from skyward.shared import codec
from skyward.worker import journal, worker
from skyward.worker.api import NotOnANodeError

pytestmark = pytest.mark.local


def describe_a_task_in_an_executor_subprocess() -> None:
    def its_print_reaches_the_journal_naming_its_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        events = tmp_path / "events.jsonl"
        monkeypatch.setattr(journal, "EVENTS", str(events))
        monkeypatch.setattr(journal, "LOCK", str(tmp_path / "events.lock"))
        monkeypatch.setattr(sys, "stdout", sys.stdout)
        monkeypatch.setattr(sys, "stderr", sys.stderr)
        monkeypatch.setenv("SKYWARD_NODE", "nod_test")
        monkeypatch.setenv("SKYWARD_COMPUTE", "cmp_test")
        monkeypatch.setenv("SKYWARD_RANK", "0")
        monkeypatch.setenv("SKYWARD_PEERS", "10.0.0.1")
        monkeypatch.setenv("SKYWARD_PLUGINS", "[]")

        def shout() -> str:
            print("<<training step 1>>")
            return "ok"

        ok, _ = worker._run_in_process("tsk_test", codec.dumps(shout), codec.dumps(((), {})), "10.0.0.1")

        assert ok
        lines = [json.loads(line) for line in events.read_text().splitlines()]
        assert {"type": "console", "content": "<<training step 1>>", "task": "tsk_test"} in lines

    def its_emit_reaches_the_journal_naming_its_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        events = tmp_path / "events.jsonl"
        monkeypatch.setattr(journal, "EVENTS", str(events))
        monkeypatch.setattr(journal, "LOCK", str(tmp_path / "events.lock"))
        monkeypatch.setattr(sys, "stdout", sys.stdout)
        monkeypatch.setattr(sys, "stderr", sys.stderr)
        monkeypatch.setenv("SKYWARD_NODE", "nod_test")
        monkeypatch.setenv("SKYWARD_COMPUTE", "cmp_test")
        monkeypatch.setenv("SKYWARD_RANK", "0")
        monkeypatch.setenv("SKYWARD_PEERS", "10.0.0.1")
        monkeypatch.setenv("SKYWARD_PLUGINS", "[]")

        def report() -> str:
            journal.emit("step", {"n": 1})
            return "ok"

        ok, _ = worker._run_in_process("tsk_test", codec.dumps(report), codec.dumps(((), {})), "10.0.0.1")

        assert ok
        assert [(event["name"], event["payload"], event["task"]) for event in emitted(events)] == [("step", {"n": 1}, "tsk_test")]


@pytest.fixture
def events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "events.jsonl"
    monkeypatch.setattr(journal, "EVENTS", str(path))
    monkeypatch.setattr(journal, "LOCK", str(tmp_path / "events.lock"))
    monkeypatch.setenv("SKYWARD_NODE", "nod_test")
    monkeypatch.setenv("SKYWARD_COMPUTE", "cmp_test")
    monkeypatch.setenv("SKYWARD_RANK", "0")
    monkeypatch.setenv("SKYWARD_PEERS", "10.0.0.1")
    return path


def consoles(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [event for event in map(json.loads, path.read_text().splitlines()) if event["type"] == "console"]


def emitted(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [event for event in map(json.loads, path.read_text().splitlines()) if event["type"] == "emitted"]


def describe_journal() -> None:
    def describe_a_partial_line() -> None:
        def it_belongs_to_the_thread_that_wrote_it(events: Path) -> None:
            out = journal.Journal("stdout")
            steps = [threading.Event() for _ in range(4)]

            def speak(name: str, turns: tuple[int, int], done: tuple[int, int]) -> None:
                journal.task.set(name)
                steps[turns[0]].wait(5)
                out.write(f"{name}-first ")
                steps[done[0]].set()
                steps[turns[1]].wait(5)
                out.write(f"{name}-second\n")
                if done[1] < len(steps):
                    steps[done[1]].set()

            first = threading.Thread(target=speak, args=("tsk_a", (0, 2), (1, 3)))
            second = threading.Thread(target=speak, args=("tsk_b", (1, 3), (2, 4)))
            first.start()
            second.start()
            steps[0].set()
            first.join(5)
            second.join(5)

            assert sorted((event["task"], event["content"]) for event in consoles(events)) == [
                ("tsk_a", "tsk_a-first tsk_a-second"),
                ("tsk_b", "tsk_b-first tsk_b-second"),
            ]

        def it_is_emitted_once_it_reaches_the_line_limit(events: Path) -> None:
            out = journal.Journal("stdout")

            out.write("x" * (journal.LINE_LIMIT - 1))
            assert consoles(events) == []

            out.write("x")
            assert [event["content"] for event in consoles(events)] == ["x" * journal.LINE_LIMIT]

        def it_stays_linear_over_many_one_character_writes(events: Path) -> None:
            out = journal.Journal("stdout")

            started = time.perf_counter()
            for _ in range(100_000):
                out.write("x")
            out.flush()
            elapsed = time.perf_counter() - started

            assert elapsed < 1.0
            assert sum(len(str(event["content"])) for event in consoles(events)) == 100_000

    def describe_append() -> None:
        def it_follows_the_paths_when_they_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
            before = tmp_path / "before.jsonl"
            after = tmp_path / "after.jsonl"
            monkeypatch.setattr(journal, "EVENTS", str(before))
            monkeypatch.setattr(journal, "LOCK", str(tmp_path / "before.lock"))
            journal.append(journal.Console(content="one"))

            monkeypatch.setattr(journal, "EVENTS", str(after))
            monkeypatch.setattr(journal, "LOCK", str(tmp_path / "after.lock"))
            journal.append(journal.Console(content="two"))

            assert [event["content"] for event in consoles(before)] == ["one"]
            assert [event["content"] for event in consoles(after)] == ["two"]

    def describe_emit() -> None:
        def it_is_a_line_the_daemon_reads_back_naming_the_running_task(events: Path) -> None:
            def running() -> None:
                journal.task.set("exe_test")
                journal.emit("epoch", {"epoch": 3, "loss": 0.25})

            before = time.time_ns() // 1_000_000
            contextvars.copy_context().run(running)

            match journal.parse(events.read_text().splitlines()[0]):
                case journal.Emitted(id=key, name="epoch", payload={"epoch": 3, "loss": 0.25}, at=at, task="exe_test"):
                    assert key and before <= at <= time.time_ns() // 1_000_000
                case other:
                    pytest.fail(f"read back as {other}")

        def every_call_is_an_event_of_its_own(events: Path) -> None:
            journal.emit("tick")
            journal.emit("tick")

            assert len({event["id"] for event in emitted(events)}) == 2

        @pytest.mark.parametrize(
            ("payload", "written"),
            [
                (None, None),
                (3, 3),
                ("done", "done"),
                ([1, 2.5, None], [1, 2.5, None]),
                ({"a": {"b": [True]}}, {"a": {"b": [True]}}),
                (MappingProxyType({"a": 1}), {"a": 1}),
                (deque([1, 2]), [1, 2]),
            ],
        )
        def it_takes_any_json_value(events: Path, payload: journal.Json, written: object) -> None:
            journal.emit("value", payload)

            assert [event["payload"] for event in emitted(events)] == [written]

        def it_refuses_what_is_not_a_json_value_at_the_call(events: Path) -> None:
            with pytest.raises(TypeError, match="object is not a JSON value"):
                journal.emit("value", {"a": object()})

            assert emitted(events) == []

        def it_refuses_an_event_over_the_line_limit(events: Path) -> None:
            with pytest.raises(ValueError, match="over the"):
                journal.emit("large", "x" * journal.LINE_LIMIT)

            assert emitted(events) == []

        def it_refuses_off_a_node(events: Path, monkeypatch: pytest.MonkeyPatch) -> None:
            monkeypatch.delenv("SKYWARD_NODE")

            with pytest.raises(NotOnANodeError):
                journal.emit("nowhere")

            assert emitted(events) == []

    def describe_parse() -> None:
        def a_metric_says_when_the_node_measured_it() -> None:
            assert journal.parse('{"type":"metric","name":"cpu","value":41.3,"at":1757887200123}') == journal.Metric(name="cpu", value=41.3, at=1757887200123)

        def a_metric_that_does_not_say_when_is_not_one() -> None:
            assert journal.parse('{"type":"metric","name":"cpu","value":41.3}') is None
