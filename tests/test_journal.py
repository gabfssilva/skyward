"""Where a subprocess's print ends up.

Thread-mode output goes through the worker's redirected stdout into the journal
the daemon tails. A process executor's child inherits a raw fd instead — the
worker's log file, which nobody streams — so the child has to redirect for
itself, and say which task was speaking.
"""

import json
import sys
from pathlib import Path

import pytest

from skyward.shared import codec
from skyward.worker import journal, worker

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
