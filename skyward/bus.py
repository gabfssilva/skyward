"""Minimal event bus for Skyward events."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from skyward.events import SkywardEvent

# Type aliases - using Any for handler types to allow specific event types
Handler = Callable[..., Any]
ReplyHandler = Callable[..., SkywardEvent | None]

# Marker attribute for handlers that return reply events
_REPLY_TO_ATTR = "_skyward_reply_to"


def respond_with[F: Callable[..., Any]](fn: F) -> F:
    """Mark handler to dispatch its return value as a new event.

    Use with @app.on() to create handlers that emit reply events:

        @app.on(Metrics)
        @app.respond_with
        def emit_cost_update(event: Metrics) -> CostUpdate:
            return CostUpdate(...)  # Dispatched automatically

    The return value (if not None) is emitted as a new event after
    the handler completes, outside the handler's context.
    """
    setattr(fn, _REPLY_TO_ATTR, True)
    return fn


class EventBus:
    """Minimal event bus for SkywardEvents."""

    def __init__(self) -> None:
        self._handlers: list[tuple[tuple[type, ...], Handler | ReplyHandler]] = []

    def on[F: Callable[..., Any]](
        self,
        *event_types: type[SkywardEvent],
    ) -> Callable[[F], F]:
        """Register handler. Empty event_types = wildcard."""

        def decorator(fn: F) -> F:
            self._handlers.append((event_types, fn))
            return fn

        return decorator

    def emit(self, event: SkywardEvent) -> None:
        """Emit event to matching handlers, dispatching replies.

        If a handler is marked with @reply_to and returns an event,
        that event is recursively emitted after the handler completes.
        """
        for types, handler in self._handlers:
            if not types or isinstance(event, types):
                result = handler(event)
                # Dispatch reply if handler is marked and returned an event
                if getattr(handler, _REPLY_TO_ATTR, False) and result is not None:
                    self.emit(result)

    def clear(self) -> None:
        """Remove all handlers."""
        self._handlers.clear()
