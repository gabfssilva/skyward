"""Application context — a lightweight console-preference holder."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

_active_app: ContextVar[App | None] = ContextVar("_active_app", default=None)


def get_app() -> App | None:
    """Return the active App instance, or ``None`` if none is active."""
    return _active_app.get()


@dataclass
class App:
    """Application context manager holding console preferences.

    Usually not needed directly — ``ComputePool`` manages its own
    session (and console) internally.

    Parameters
    ----------
    console
        Whether to enable Rich console output. Default ``True``.

    Examples
    --------
    >>> with sky.App(console=True):
    ...     with sky.ComputePool(...) as compute:
    ...         result = train(data) >> compute
    """
    console: bool = True

    _context_token: Token[App | None] | None = field(default=None, init=False, repr=False)

    def __enter__(self) -> App:
        self._context_token = _active_app.set(self)
        return self

    def __exit__(self, *args: Any) -> None:
        if self._context_token is not None:
            _active_app.reset(self._context_token)
            self._context_token = None
