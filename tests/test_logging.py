"""The logger writes off the logging thread, and leaves the host's logging alone."""

import importlib
import logging
import logging.handlers
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from skyward.server import daemon
from skyward.shared.observability.logger import NAME, logger

module = importlib.import_module("skyward.shared.observability.logger")

pytestmark = pytest.mark.local


@pytest.fixture(autouse=True)
def isolated() -> Iterator[None]:
    saved = dict(module._handlers)
    logger.remove()
    yield
    logger.remove()
    module._handlers.update(saved)
    module._rewire()


def _skyward_handlers(target: logging.Logger) -> list[logging.Handler]:
    return [handler for handler in target.handlers if handler is module._front or handler in module._handlers.values()]


def describe_file_sink() -> None:
    def it_writes_records_from_a_thread_other_than_the_one_that_logged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        writers: set[int] = set()
        original = logging.handlers.RotatingFileHandler.emit

        def emit(self: logging.handlers.RotatingFileHandler, record: logging.LogRecord) -> None:
            writers.add(threading.get_ident())
            original(self, record)

        monkeypatch.setattr(logging.handlers.RotatingFileHandler, "emit", emit)
        sink = logger.add(str(tmp_path / "skyward.log"))
        logger.info("hello")
        logger.remove(sink)

        assert writers
        assert threading.get_ident() not in writers

    def it_has_written_every_record_once_the_sink_is_removed(tmp_path: Path) -> None:
        path = tmp_path / "skyward.log"
        sink = logger.add(str(path))
        for index in range(2000):
            logger.info("record {index}", index=index)
        logger.remove(sink)

        lines = path.read_text().splitlines()
        assert len(lines) == 2000
        assert lines[-1].endswith("record 1999")


def describe_without_sinks() -> None:
    def it_attaches_nothing_to_the_root_logger(tmp_path: Path) -> None:
        sink = logger.add(str(tmp_path / "skyward.log"))
        logger.remove(sink)

        assert _skyward_handlers(logging.getLogger()) == []
        assert _skyward_handlers(logging.getLogger(NAME)) == []
        assert module._listener is None

    def it_leaves_no_thread_writing(tmp_path: Path) -> None:
        before = set(threading.enumerate())
        sink = logger.add(str(tmp_path / "skyward.log"))
        logger.remove(sink)

        assert set(threading.enumerate()) - before == set()


def describe_spawn() -> None:
    def it_starts_the_daemon_without_an_access_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        commands: list[list[str]] = []

        class Process:
            pid = 4242

        def popen(command: list[str], **_: object) -> Process:
            commands.append(command)
            return Process()

        monkeypatch.setattr(daemon, "RUNTIME_DIR", tmp_path)
        monkeypatch.setattr(daemon, "LOG_FILE", tmp_path / "server.log")
        monkeypatch.setattr(daemon, "installed", lambda: True)
        monkeypatch.setattr(subprocess, "Popen", popen)

        assert daemon.spawn("127.0.0.1", 17999) == 4242
        assert len(commands) == 1
        assert "--no-access-log" in commands[0]
