from __future__ import annotations

from collections.abc import Callable

from casty import Behavior

from skyward.actors.console.actor import console_actor
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
    "minimal_console_actor",
    "resolve_console",
]


def resolve_console(
    mode: bool | ConsoleMode,
) -> Callable[[], Behavior[ConsoleInput]] | None:
    """Map a console mode to its actor factory.

    Parameters
    ----------
    mode
        Either a legacy ``bool`` (``True`` → rich, ``False`` → silent)
        or a ``ConsoleMode`` literal (``"rich"``, ``"minimal"``, ``"silent"``).

    Returns
    -------
    Callable[[], Behavior[ConsoleInput]] | None
        Factory that constructs the chosen console behavior, or ``None``
        when no console should be spawned.
    """
    match mode:
        case True | "rich":
            return console_actor
        case "minimal":
            return minimal_console_actor
        case False | "silent":
            return None
