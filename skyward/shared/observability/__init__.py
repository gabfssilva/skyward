"""Structured logging, standard library only.

A node installs skyward to run a function; it should not acquire a logging
framework to do it. So the logger here is hand-rolled over ``logging`` — loguru's
shape, none of its weight, and no console dependency: ``rich`` is an opt-in extra
in this package and a sink that needed it would not work where it is absent.

What a daemon logs is kept in a log file — JSON lines, numbered — that it reads back
to answer for its own log.

Metrics live in ``skyward.worker.metrics`` (the public ``sky.metrics`` namespace) and are
deliberately not re-exported here — one name, one import path.
"""

from skyward.shared.observability.logfile import Entry, Failure, Group, LogFile, Query, Severity, Summary, Volume, entries, summarize
from skyward.shared.observability.logger import NAME, Logger, logger
from skyward.shared.observability.logging import Installed, LogConfig, LogLevel, level, setup_logging, teardown_logging
from skyward.shared.observability.notice import notice

__all__ = [
    "NAME",
    "Entry",
    "Failure",
    "Group",
    "Installed",
    "LogConfig",
    "LogFile",
    "LogLevel",
    "Logger",
    "Query",
    "Severity",
    "Summary",
    "Volume",
    "entries",
    "level",
    "logger",
    "notice",
    "setup_logging",
    "summarize",
    "teardown_logging",
]
