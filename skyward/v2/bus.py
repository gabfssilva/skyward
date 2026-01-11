"""AsyncEventBus - async event routing with blinker.

Thin wrapper around blinker's native async support, adding:
- emit() - fire-and-forget (creates tasks, doesn't block)
- request() - request/response correlation via request_id

For simple await-style handlers, use blinker's send_async() directly.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from blinker import Signal


# =============================================================================
# AsyncEventBus
# =============================================================================


class AsyncEventBus:
    """
    Async event bus using blinker's native async support.

    Methods:
    - emit(event) - fire and forget, creates tasks without blocking
    - emit_await(event) - await all handlers (uses blinker's send_async)
    - request(command, response_type) - emit and wait for correlated response
    - connect(event_type, handler) - register async handler
    - drain() - wait for pending fire-and-forget tasks
    """

    def __init__(self) -> None:
        self._signals: dict[type, Signal] = {}
        self._pending: set[asyncio.Task[Any]] = set()
        self._waiters: dict[str, asyncio.Future[Any]] = {}
        self._shutdown = False

    def _signal_for(self, event_type: type) -> Signal:
        """Get or create signal for event type."""
        if event_type not in self._signals:
            self._signals[event_type] = Signal(event_type.__name__)
        return self._signals[event_type]

    def connect(self, event_type: type, handler: Callable[..., Any]) -> None:
        """Register async handler for event type.

        Uses strong references (weak=False) to prevent handlers from being
        garbage collected while still registered.
        """
        self._signal_for(event_type).connect(handler, weak=False)

    def disconnect(self, event_type: type, handler: Callable[..., Any]) -> None:
        """Unregister handler for event type."""
        signal = self._signals.get(event_type)
        if signal:
            signal.disconnect(handler)

    def emit(self, event: object) -> asyncio.Task[list[Any]] | None:
        """
        Emit event, fire-and-forget.

        Creates a task that runs all handlers but doesn't block.
        Returns the task (can be ignored for true fire-and-forget).
        """
        if self._shutdown:
            raise RuntimeError("EventBus has been shut down")

        signal = self._signals.get(type(event))
        if not signal or not signal.receivers:
            self._check_waiter(event)
            return None

        async def run_handlers() -> list[Any]:
            # Use blinker's native send_async
            results = await signal.send_async(self, event=event)
            self._check_waiter(event)
            return [r for _, r in results]

        task = asyncio.create_task(run_handlers())
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

        return task

    async def emit_await(self, event: object) -> list[Any]:
        """
        Emit event and wait for all handlers to complete.

        Uses blinker's native send_async() directly.

        Returns:
            List of results from handlers.
        """
        if self._shutdown:
            raise RuntimeError("EventBus has been shut down")

        signal = self._signals.get(type(event))
        if not signal or not signal.receivers:
            self._check_waiter(event)
            return []

        # Use blinker's native send_async
        results = await signal.send_async(self, event=event)
        self._check_waiter(event)

        return [r for _, r in results]

    async def request[T](
        self,
        command: object,
        response_type: type[T],
        timeout: float = 300.0,
        match: Callable[[T], bool] | None = None,
    ) -> T:
        """
        Emit command and wait for correlated response.

        Uses request_id field for correlation.

        Args:
            command: Command event (must have request_id field)
            response_type: Type of response event to wait for
            timeout: Maximum wait time
            match: Optional predicate to filter responses

        Returns:
            Response event
        """
        request_id = getattr(command, "request_id", None)
        if request_id is None:
            raise ValueError(f"Command {type(command).__name__} must have request_id field")

        # Create future for response
        loop = asyncio.get_running_loop()
        future: asyncio.Future[T] = loop.create_future()
        waiter_key = f"{response_type.__name__}:{request_id}"
        self._waiters[waiter_key] = future

        # Store match predicate if provided
        if match:
            future.__match_predicate__ = match  # type: ignore[attr-defined]

        try:
            # Emit command (fire-and-forget)
            self.emit(command)

            # Wait for response
            return await asyncio.wait_for(future, timeout=timeout)

        finally:
            self._waiters.pop(waiter_key, None)

    def _check_waiter(self, event: object) -> None:
        """Check if event satisfies any waiting request."""
        request_id = getattr(event, "request_id", None)
        if request_id is None:
            return

        waiter_key = f"{type(event).__name__}:{request_id}"
        future = self._waiters.get(waiter_key)

        if future and not future.done():
            match_fn = getattr(future, "__match_predicate__", None)
            if match_fn and not match_fn(event):
                return
            future.set_result(event)

    async def drain(self, timeout: float | None = None) -> None:
        """Wait for all pending fire-and-forget tasks."""
        if not self._pending:
            return

        pending = list(self._pending)
        _, not_done = await asyncio.wait(
            pending,
            timeout=timeout,
            return_when=asyncio.ALL_COMPLETED,
        )

        if not_done:
            raise asyncio.TimeoutError(f"{len(not_done)} tasks still pending")

    async def shutdown(self, wait: bool = True, timeout: float = 30.0) -> None:
        """Shutdown the event bus gracefully."""
        self._shutdown = True

        if wait:
            try:
                await self.drain(timeout=timeout)
            except asyncio.TimeoutError:
                for task in self._pending:
                    task.cancel()

        for future in self._waiters.values():
            if not future.done():
                future.cancel()

        self._waiters.clear()

    @property
    def pending_count(self) -> int:
        """Number of fire-and-forget tasks still running."""
        return len(self._pending)

    @property
    def is_shutdown(self) -> bool:
        """Whether bus has been shut down."""
        return self._shutdown


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "AsyncEventBus",
]
