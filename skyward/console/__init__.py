from __future__ import annotations

import os
import sys
from collections.abc import Callable

from skyward.api.spec import ConsoleMode
from skyward.console.consumer import ConsoleConsumer, Renderer
from skyward.console.log import LogConsole
from skyward.console.messages import (
    ConsoleInput,
    EventReceived,
    LocalOutput,
    LogReceived,
    ViewUpdated,
)
from skyward.console.minimal import MinimalConsole
from skyward.console.renderer import RichConsole

__all__ = [
    "ConsoleConsumer",
    "ConsoleInput",
    "ConsoleMode",
    "EventReceived",
    "LocalOutput",
    "LogConsole",
    "LogReceived",
    "MinimalConsole",
    "Renderer",
    "RichConsole",
    "ViewUpdated",
    "resolve_console",
]


def _is_tty() -> bool:
    """Return True when stderr looks like an interactive terminal.

    Honors ``SKYWARD_CONSOLE_FORCE_TTY`` (``1``/``true``/``yes``) so users
    can force-enable rich/minimal output even when stderr is piped — useful
    for debugging the Live renderers.
    """
    override = os.environ.get("SKYWARD_CONSOLE_FORCE_TTY", "").strip().lower()
    if override in {"1", "true", "yes"}:
        return True
    stderr = sys.stderr
    return bool(stderr and hasattr(stderr, "isatty") and stderr.isatty())


def resolve_console(mode: bool | ConsoleMode) -> Callable[[], Renderer] | None:
    """Map a console mode to its renderer factory.

    ``rich`` and ``minimal`` require a TTY on stderr; when stderr is a
    pipe, file, CI log, or otherwise non-interactive, both fall back to
    ``log``.  ``log`` and ``silent`` are honored unconditionally.
    """
    match mode:
        case True | "rich":
            return RichConsole if _is_tty() else LogConsole
        case "minimal":
            return MinimalConsole if _is_tty() else LogConsole
        case "log":
            return LogConsole
        case False | "silent":
            return None
