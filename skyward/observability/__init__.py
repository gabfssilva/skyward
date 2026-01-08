"""Observability module - logging and output control.

Contains logging configuration and output control.
"""

from skyward.observability.logging import LogConfig, _setup_logging, _teardown_logging
from skyward.observability.output import (
    CallbackWriter,
    is_head,
    redirect_output,
    silent,
    stderr,
    stdout,
)

__all__ = [
    # Logging
    "LogConfig",
    "_setup_logging",
    "_teardown_logging",
    # Output
    "stdout",
    "stderr",
    "silent",
    "is_head",
    "CallbackWriter",
    "redirect_output",
]
