"""Turning a ``LogConfig`` into sinks, and taking them away again.

``setup_logging`` is the whole configuration surface: a level, a console, a file
with a rotation policy. It returns the handler ids ``teardown_logging`` needs,
so a caller can enable logging for the length of a block and leave the process's
logging exactly as it found it::

    from skyward.observability import LogConfig, setup_logging, teardown_logging

    ids = setup_logging(LogConfig(level="DEBUG", file="skyward.log"))
    ...
    teardown_logging(ids)

Both are idempotent: ``setup_logging`` drops the sinks it installed last time
before installing new ones, and removing an id twice is a no-op.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from skyward.observability.logger import NAME, logger

type LogLevel = Literal["TRACE", "DEBUG", "INFO", "WARNING", "ERROR"]
"""Supported log severity levels, finest first."""

_CONTEXT_KEYS = ("component", "integration", "provider", "compute_id", "node_id", "instance_id", "collection", "name")


@dataclass(frozen=True, slots=True)
class LogConfig:
    """What to log, and where.

    Parameters
    ----------
    level
        Minimum severity the console accepts. The file always takes ``DEBUG``.
    file
        Path to the log file. Default ``~/.skyward/logs/skyward.log``.
    console
        Whether to also log to stdout.
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


def _context(record: logging.LogRecord) -> str:
    extras: dict[str, object] = getattr(record, "extras", {})
    parts = [f"{key}={extras[key]}" for key in _CONTEXT_KEYS if key in extras]
    return f" [{' '.join(parts)}]" if parts else ""


def _patcher(record: logging.LogRecord) -> None:
    record.__dict__["_ctx"] = _context(record)


def setup_logging(config: LogConfig) -> list[int]:
    """Install the configured sinks and return their ids, for ``teardown_logging``."""
    logger.remove()
    logger.enable()
    logger.configure(patcher=_patcher)

    ids: list[int] = []

    if config.console:
        ids.append(logger.add(sys.stdout, level=config.level, filter=NAME))

    if config.file:
        Path(config.file).parent.mkdir(parents=True, exist_ok=True)
        ids.append(
            logger.add(
                config.file,
                level="DEBUG",
                rotation=config.rotation,
                retention=config.retention,
                compression="gzip",
            )
        )

    return ids


def teardown_logging(ids: list[int]) -> None:
    """Remove the sinks ``setup_logging`` installed and silence the logger."""
    for handler_id in ids:
        logger.remove(handler_id)
    logger.disable()
