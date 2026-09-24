"""Turning a ``LogConfig`` into sinks, and taking them away again.

``setup_logging`` is the whole configuration surface: a level, a console, a log
file with a rotation policy. It returns what it installed — the sink ids
``teardown_logging`` needs, and the log file, for whoever reads it back — so a
caller can enable logging for the length of a block and leave the process's logging
exactly as it found it::

    from skyward.shared.observability import LogConfig, setup_logging, teardown_logging

    installed = setup_logging(LogConfig(level="DEBUG", file="skyward.log"))
    ...
    teardown_logging(installed.ids)

Both are idempotent: ``setup_logging`` drops the sinks it installed last time
before installing new ones, and removing an id twice is a no-op.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple

from skyward.shared.observability.logfile import LogFile
from skyward.shared.observability.logger import NAME, logger

type LogLevel = Literal["TRACE", "DEBUG", "INFO", "WARNING", "ERROR"]
"""Supported log severity levels, finest first."""

CAPTURED = ("litestar", "casty", "asyncssh", "uvicorn.error")
"""The libraries whose warnings and errors belong in the log file beside the daemon's own lines.

Left alone, a library logger with nothing configured falls through to
``logging.lastResort``: stderr, which for a detached daemon is a file nothing
rotates, in lines with no time on them.
"""

_CONTEXT_KEYS = ("component", "integration", "provider", "compute_id", "node_id", "instance_id", "collection", "name")


@dataclass(frozen=True, slots=True)
class LogConfig:
    """What to log, and where.

    Parameters
    ----------
    level
        Minimum severity the console accepts. The file always takes ``DEBUG``.
    file
        Path to the log file, one JSON entry per line. Default ``~/.skyward/logs/skyward.log``.
    console
        Whether to also log to stdout, as text.
    rotation
        Size at which the file rolls over.
    retention
        How many rolled files to keep.
    """

    level: LogLevel = "INFO"
    file: str = str(Path.home() / ".skyward" / "logs" / "skyward.log")
    console: bool = True
    rotation: str = "50 MB"
    retention: int = 10


class Installed(NamedTuple):
    """What ``setup_logging`` put in place: the sinks to remove, and the log file, when there is one."""

    ids: list[int]
    file: LogFile | None


def _context(record: logging.LogRecord) -> str:
    extras: dict[str, object] = getattr(record, "extras", {})
    parts = [f"{key}={extras[key]}" for key in _CONTEXT_KEYS if key in extras]
    return f" [{' '.join(parts)}]" if parts else ""


def _patcher(record: logging.LogRecord) -> None:
    record.__dict__["_ctx"] = _context(record)


def level(name: str | None) -> LogLevel:
    """The level a name asks for, or ``INFO`` when it names none this understands.

    The console is the only sink a level applies to — the file always takes
    ``DEBUG``, because the thing worth reading after a failure is the detail
    nobody wanted on their terminal while it was happening.
    """
    match (name or "").strip().upper():
        case "TRACE":
            return "TRACE"
        case "DEBUG":
            return "DEBUG"
        case "WARNING":
            return "WARNING"
        case "ERROR":
            return "ERROR"
        case _:
            return "INFO"


def setup_logging(config: LogConfig) -> Installed:
    """Install the configured sinks, and have the libraries in :data:`CAPTURED` speak through them."""
    logger.remove()
    logger.enable()
    logger.configure(patcher=_patcher)

    ids: list[int] = []

    if config.console:
        ids.append(logger.add(sys.stdout, level=config.level, filter=NAME))

    file = LogFile(Path(config.file), size=_bytes(config.rotation), keep=config.retention) if config.file else None
    if file is not None:
        ids.append(logger.add(file, level="DEBUG"))

    for name in CAPTURED:
        logger.capture(name)

    return Installed(ids, file)


def teardown_logging(ids: list[int]) -> None:
    """Remove the sinks ``setup_logging`` installed, let the libraries go, and silence the logger."""
    for handler_id in ids:
        logger.remove(handler_id)
    for name in CAPTURED:
        logger.release(name)
    logger.disable()


def _bytes(rotation: str) -> int:
    match rotation.strip().split():
        case [size, unit] if unit.upper() == "MB":
            return int(size) * 1024 * 1024
        case _:
            return 50 * 1024 * 1024
