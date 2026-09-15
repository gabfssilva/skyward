"""The logger writes off the logging thread, and leaves the host's logging alone."""

import importlib
import logging
import logging.handlers
import os
import signal
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path

import httpx
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
    @pytest.mark.timeout(60)
    def it_starts_a_daemon_that_answers_and_writes_no_access_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        monkeypatch.setattr(daemon, "RUNTIME_DIR", tmp_path)
        monkeypatch.setattr(daemon, "LOG_FILE", tmp_path / "server.log")
        monkeypatch.setenv("HOME", str(tmp_path))

        process = daemon.spawn("127.0.0.1", port, tmp_path / "skyward.sqlite")
        try:
            assert _answered(f"http://127.0.0.1:{port}/v1/health/live"), "the detached daemon came up"
        finally:
            os.kill(process, signal.SIGTERM)
            with suppress(ChildProcessError):
                os.waitpid(process, 0)

        assert "GET /v1/health/live" not in (tmp_path / "server.log").read_text(), "a line per request is noise in a file nothing rotates"


def _answered(url: str) -> bool:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        with suppress(httpx.HTTPError):
            if httpx.get(url, timeout=1).status_code == 200:
                return True
        time.sleep(0.2)
    return False
