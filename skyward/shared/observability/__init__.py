"""Structured logging, standard library only.

A node installs skyward to run a function; it should not acquire a logging
framework to do it. So the logger here is hand-rolled over ``logging`` — loguru's
shape, none of its weight, and no console dependency: ``rich`` is an opt-in extra
in this package and a sink that needed it would not work where it is absent.

Metrics live in ``skyward.worker.metrics`` (the public ``sky.metrics`` namespace) and are
deliberately not re-exported here — one name, one import path.
"""

from skyward.shared.observability.logger import NAME, Logger, logger
from skyward.shared.observability.logging import LogConfig, LogLevel, level, setup_logging, teardown_logging

__all__ = [
    "NAME",
    "LogConfig",
    "LogLevel",
    "Logger",
    "level",
    "logger",
    "setup_logging",
    "teardown_logging",
]
