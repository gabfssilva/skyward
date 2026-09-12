"""``sky server`` and the pid this machine recorded: what stop and restart do to it."""

from __future__ import annotations

import pytest

from skyward.cli import server
from skyward.server import daemon

pytestmark = pytest.mark.local


class Recorded:
    """The daemon this machine recorded, as the commands are able to see it."""

    def __init__(self, process: int | None) -> None:
        self.process = process
        self.signalled: list[int] = []
        self.started = 0

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(daemon, "pid", lambda: self.process)
        monkeypatch.setattr(daemon, "alive", lambda process: self.process == process)
        monkeypatch.setattr(daemon, "forget", self.forget)
        monkeypatch.setattr(server.os, "kill", self.kill)
        monkeypatch.setattr(server, "start", self.start)
        monkeypatch.setattr(server, "live", lambda target: False)

    def kill(self, process: int, _signal: int) -> None:
        self.signalled.append(process)
        self.process = None

    def forget(self) -> None:
        self.process = None

    def start(self, **_: object) -> None:
        self.started += 1


def describe_restarting_the_daemon() -> None:
    def it_stops_the_one_this_machine_started_before_it_starts_another(monkeypatch: pytest.MonkeyPatch) -> None:
        recorded = Recorded(4242)
        recorded.install(monkeypatch)

        server.restart()

        assert recorded.signalled == [4242], "the pid was signalled, not asked"
        assert recorded.started == 1

    def it_starts_one_when_there_is_nothing_to_stop(monkeypatch: pytest.MonkeyPatch) -> None:
        recorded = Recorded(None)
        recorded.install(monkeypatch)

        server.restart()

        assert recorded.signalled == [] and recorded.started == 1

    def it_refuses_when_something_answers_that_this_machine_did_not_start(monkeypatch: pytest.MonkeyPatch) -> None:
        recorded = Recorded(None)
        recorded.install(monkeypatch)
        monkeypatch.setattr(server, "live", lambda target: True)

        with pytest.raises(SystemExit) as refused:
            server.restart()

        assert "no pid" in str(refused.value)
        assert recorded.started == 0, "a daemon somebody else started is not restarted behind their back"


def describe_stopping_the_daemon() -> None:
    def it_says_there_is_nothing_to_stop_when_no_pid_was_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
        Recorded(None).install(monkeypatch)

        with pytest.raises(SystemExit) as refused:
            server.stop()

        assert "nothing to stop" in str(refused.value)

    def it_clears_a_pid_whose_process_is_already_gone(monkeypatch: pytest.MonkeyPatch) -> None:
        recorded = Recorded(4242)
        recorded.install(monkeypatch)
        monkeypatch.setattr(daemon, "alive", lambda process: False)

        server.stop()

        assert recorded.signalled == [] and recorded.process is None
