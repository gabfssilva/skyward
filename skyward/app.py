"""Application context — owns console lifecycle and spy wiring."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

from casty import ActorRef, ActorSystem

from skyward.observability.logger import logger

_active_app: ContextVar[App | None] = ContextVar("_active_app", default=None)


def get_app() -> App | None:
    """Return the active App instance, or ``None`` if none is active."""
    return _active_app.get()


@dataclass
class App:
    """Application context manager for console lifecycle and spy wiring.

    Provide a Rich adaptive console and optional spy actor for
    observing pool events. Usually not needed directly — ``ComputePool``
    manages its own ``App`` internally.

    Parameters
    ----------
    console
        Whether to enable Rich console output. Default ``True``.

    Examples
    --------
    >>> with sky.App(console=True):
    ...     with sky.ComputePool(...) as pool:
    ...         result = train(data) >> pool
    """
    console: bool = True

    _spy_ref: ActorRef | None = field(default=None, init=False, repr=False)
    _context_token: Token[App | None] | None = field(default=None, init=False, repr=False)

    @property
    def spy(self) -> ActorRef | None:
        return self._spy_ref

    def __enter__(self) -> App:
        self._context_token = _active_app.set(self)
        return self

    def __exit__(self, *args: Any) -> None:
        if self._context_token is not None:
            _active_app.reset(self._context_token)
            self._context_token = None

    def setup(self, system: ActorSystem, spec: Any) -> None:
        """Wire App into the actor system. Spawns console if enabled."""
        if not self.console or self._spy_ref is not None:
            return

        from skyward.actors.console import console_actor

        self._spy_ref = system.spawn(console_actor(spec), "console")
        logger.debug("Console actor spawned")
