"""Logging configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "TRACE"]


@dataclass(frozen=True, slots=True)
class LogConfig:
    """Logging configuration for ComputePool.

    Parameters
    ----------
    level
        Minimum log level.
    file
        Path to log file.
    console
        Whether to log to stderr.
    rotation
        File rotation policy (e.g., "50 MB", "1 day").
    retention
        Number of old log files to keep.
    """

    level: LogLevel = "INFO"
    file: str = str(Path.home() / ".skyward" / "logs" / "skyward.log")
    console: bool = True
    rotation: str = "50 MB"
    retention: int = 10
