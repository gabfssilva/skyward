"""Callback-based event system for Skyward.

This module provides a simple, deadlock-safe event dispatch mechanism.
Events are processed asynchronously in a background thread to prevent
blocking the caller.

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

import atexit
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from queue import Queue
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skyward.events import SkywardEvent

type CallbackResult = SkywardEvent | Sequence[SkywardEvent] | None
type Callback = Callable[[SkywardEvent], CallbackResult]

_callback: ContextVar[Callback | None] = ContextVar("skyward_cb", default=None)

# Global event queue and worker
_event_queue: Queue[tuple[Callback, SkywardEvent] | None] = Queue()
_worker_thread: threading.Thread | None = None
_worker_lock = threading.Lock()


def _normalize(result: CallbackResult) -> list[SkywardEvent]:
    """Convert callback result to list of events."""
    match result:
        case None:
            return []
        case [*events]:
            return list(events)
        case event:
            return [event]


def _worker_loop() -> None:
    """Background worker that processes events."""
    while True:
        item = _event_queue.get()
        if item is None:  # Shutdown signal
            break
        cb, event = item
        try:
            derived = cb(event)
            for e in _normalize(derived):
                _event_queue.put((cb, e))
        except Exception:
            pass  # Don't crash worker on callback errors
        finally:
            _event_queue.task_done()


def _ensure_worker() -> None:
    """Start worker thread if not already running."""
    global _worker_thread
    with _worker_lock:
        if _worker_thread is None or not _worker_thread.is_alive():
            _worker_thread = threading.Thread(target=_worker_loop, daemon=True)
            _worker_thread.start()


def _shutdown_worker() -> None:
    """Gracefully shutdown worker thread."""
    if _worker_thread is not None and _worker_thread.is_alive():
        _event_queue.put(None)  # Shutdown signal
        _worker_thread.join(timeout=2.0)


# Register shutdown handler
atexit.register(_shutdown_worker)


def emit(event: SkywardEvent) -> None:
    """Emit event to the current context's callback (non-blocking).

    The event is queued and processed asynchronously in a background thread.
    This prevents slow callbacks from blocking the caller.

    Args:
        event: The event to emit.
    """
    cb = _callback.get()
    if cb is None:
        return

    _ensure_worker()
    _event_queue.put((cb, event))


def flush() -> None:
    """Wait for all queued events to be processed.

    Useful for testing or when you need to ensure all events
    have been handled before proceeding.
    """
    _event_queue.join()


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


__all__ = [
    "Callback",
    "CallbackResult",
    "emit",
    "flush",
    "compose",
    "use_callback",
]
