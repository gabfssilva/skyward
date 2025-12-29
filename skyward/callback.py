"""Callback-based event system for Skyward.

This module provides a simple, deadlock-safe event dispatch mechanism.
Callbacks receive events and may return derived events, which are
processed in a queue to ensure locks are released between dispatches.

Example:
    from skyward.callback import emit, use_callback, compose

    def my_callback(event):
        match event:
            case Metrics(gpu_utilization=gpu) if gpu > 90:
                print(f"GPU hot: {gpu}%")

    with use_callback(my_callback):
        emit(Metrics(...))
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skyward.events import SkywardEvent

type CallbackResult = SkywardEvent | Sequence[SkywardEvent] | None
type Callback = Callable[[SkywardEvent], CallbackResult]

_callback: ContextVar[Callback | None] = ContextVar("skyward_cb", default=None)


def _normalize(result: CallbackResult) -> list[SkywardEvent]:
    """Convert callback result to list of events."""
    if result is None:
        return []
    # Check for Sequence but exclude string (which is Sequence[str])
    if isinstance(result, Sequence) and not isinstance(result, str):
        return list(result)
    return [result]  # type: ignore[list-item]


def emit(event: SkywardEvent) -> None:
    """Emit event to the current context's callback.

    Processes derived events in a queue (breadth-first), ensuring
    that any locks held by callbacks are released between dispatches.

    This design prevents deadlocks when callbacks return derived events:
    1. Callback is called with event
    2. Callback acquires lock, computes, releases lock, returns derived events
    3. Derived events are added to queue
    4. Next event is processed (no locks held)

    Args:
        event: The event to emit.
    """
    cb = _callback.get()
    if cb is None:
        return

    queue: list[SkywardEvent] = [event]
    while queue:
        current = queue.pop(0)
        derived = cb(current)
        queue.extend(_normalize(derived))


def compose(*callbacks: Callback) -> Callback:
    """Combine multiple callbacks into one.

    Each callback receives every event. Derived events from all
    callbacks are collected and returned together.

    Args:
        *callbacks: Callbacks to compose.

    Returns:
        A single callback that dispatches to all provided callbacks.

    Example:
        combined = compose(log_callback, cost_tracker, user_callback)
        with use_callback(combined):
            emit(event)
    """
    match callbacks:
        case []:
            return lambda _: None
        case [single]:
            return single
        case _:

            def combined(event: SkywardEvent) -> list[SkywardEvent]:
                results: list[SkywardEvent] = []
                for cb in callbacks:
                    results.extend(_normalize(cb(event)))
                return results

            return combined


@contextmanager
def use_callback(cb: Callback) -> Iterator[None]:
    """Context manager that sets the active callback.

    The callback will receive all events emitted via emit() within
    this context. Callbacks can return derived events which will
    be dispatched after the callback returns (ensuring locks are released).

    Args:
        cb: The callback to activate.

    Example:
        with use_callback(my_handler):
            emit(SomeEvent())  # my_handler receives this
    """
    token = _callback.set(cb)
    try:
        yield
    finally:
        _callback.reset(token)


def only(*event_types: type[SkywardEvent]) -> Callable[[Callback], Callback]:
    """Decorator that filters a callback to only receive specific event types.

    Args:
        *event_types: Event types to pass through.

    Returns:
        A decorator that wraps the callback with filtering.

    Example:
        @only(Error, BootstrapCompleted)
        def my_callback(event):
            print(event)  # Only receives Error and BootstrapCompleted
    """

    def decorator(cb: Callback) -> Callback:
        def filtered(event: SkywardEvent) -> CallbackResult:
            if isinstance(event, event_types):
                return cb(event)
            return None

        return filtered

    return decorator


__all__ = [
    "Callback",
    "CallbackResult",
    "emit",
    "compose",
    "use_callback",
    "only",
]
