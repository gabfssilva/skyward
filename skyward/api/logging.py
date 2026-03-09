"""Logging configuration for Skyward sessions and pools.

Controls log level, output destinations, and file rotation policy.
Pass a ``LogConfig`` instance (or ``True`` for defaults, ``False`` to
disable) to ``Options.logging`` or ``Session(logging=...)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "TRACE"]
"""Supported log severity levels.

- ``"TRACE"`` — finest-grained internal diagnostics.
- ``"DEBUG"`` — detailed diagnostic information.
- ``"INFO"`` — confirmation that things are working as expected.
- ``"WARNING"`` — something unexpected but non-fatal.
- ``"ERROR"`` — a task or subsystem failed.
"""


@dataclass(frozen=True, slots=True)
class LogConfig:
    """Logging configuration for sessions and pools.

    Sensible defaults are provided — most users only need ``True``
    (use defaults) or ``False`` (disable) via ``Options.logging``.

    Parameters
    ----------
    level
        Minimum severity level to emit.  Default ``"INFO"``.
    file
        Absolute path to the log file.
        Default ``~/.skyward/logs/skyward.log``.
    console
        Whether to also log to stderr.  Default ``True``.
    rotation
        File rotation policy.  Accepts size-based (``"50 MB"``) or
        time-based (``"1 day"``) policies.  Default ``"50 MB"``.
    retention
        Number of rotated log files to keep before deletion.
        Default ``10``.
    """

    level: LogLevel = "INFO"
    file: str = str(Path.home() / ".skyward" / "logs" / "skyward.log")
    console: bool = True
    rotation: str = "50 MB"
    retention: int = 10
