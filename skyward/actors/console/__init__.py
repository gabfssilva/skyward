from __future__ import annotations

import os
import sys
from collections.abc import Callable

from casty import Behavior

from skyward.actors.console.actor import console_actor
from skyward.actors.console.log import log_console_actor
from skyward.actors.console.messages import (
    ConsoleInput,
    EventReceived,
    LocalOutput,
    LogReceived,
    ViewUpdated,
)
from skyward.actors.console.minimal import minimal_console_actor
from skyward.api.spec import ConsoleMode

__all__ = [
    "ConsoleInput",
    "ConsoleMode",
    "EventReceived",
    "LocalOutput",
    "LogReceived",
    "ViewUpdated",
    "console_actor",
    "log_console_actor",
    "minimal_console_actor",
    "print_no_offers_error",
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


def resolve_console(
    mode: bool | ConsoleMode,
) -> Callable[[], Behavior[ConsoleInput]] | None:
    """Map a console mode to its actor factory.

    ``rich`` and ``minimal`` require a TTY on stderr; when stderr is a
    pipe, file, CI log, or otherwise non-interactive, both fall back to
    ``log``.  ``log`` and ``silent`` are honored unconditionally.

    Parameters
    ----------
    mode
        Either a legacy ``bool`` (``True`` → rich, ``False`` → silent)
        or a ``ConsoleMode`` literal (``"rich"``, ``"minimal"``,
        ``"log"``, ``"silent"``).

    Returns
    -------
    Callable[[], Behavior[ConsoleInput]] | None
        Factory that constructs the chosen console behavior, or ``None``
        when no console should be spawned.
    """
    match mode:
        case True | "rich":
            return console_actor if _is_tty() else log_console_actor
        case "minimal":
            return minimal_console_actor if _is_tty() else log_console_actor
        case "log":
            return log_console_actor
        case False | "silent":
            return None


def print_no_offers_error(error: BaseException, mode: bool | ConsoleMode) -> None:
    """Render a *no matching offers* panel to stderr when a console is enabled.

    Offer selection fails before any pool or console actor exists, so this is
    a synchronous render at the startup boundary rather than an event-driven
    one. No-op when the console is disabled (``silent``/``False``) — headless
    callers let the exception propagate instead of presenting it.

    Parameters
    ----------
    error
        The ``NoOffersError`` to render (ignored if a different type).
    mode
        The session console mode, used to decide whether to render.
    """
    if resolve_console(mode) is None:
        return
    from rich.console import Console

    from skyward.actors.console.view import _print_no_offers_error

    _print_no_offers_error(Console(stderr=True), error)
