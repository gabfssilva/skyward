from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

from litestar.events import BaseEventEmitterBackend, EventListener

logger = logging.getLogger(__name__)

type Key = tuple[EventListener, str, int]


class ReconcilingEventEmitter(BaseEventEmitterBackend):
    """In-process emitter with per-key coalescing and error isolation.

    Identical ``(listener, event, payload)`` triples never run concurrently: a
    duplicate emitted while one is in flight marks the key dirty and re-runs it
    exactly once when the current run finishes. That is what makes an event a
    wakeup rather than a unit of work — N wakeups for the same compute collapse
    into one reconcile, and the last one always wins.

    Unhashable payloads (notifications carrying blobs or dicts) are dispatched
    concurrently, the way ``SimpleEventEmitter`` does: they are data, not
    triggers, so there is nothing to coalesce.

    A listener that raises never escapes into a task group. Under
    ``SimpleEventEmitter`` a single failing handler kills the worker and every
    subsequent event is dropped silently.
    """

    def __init__(self, listeners: Sequence[EventListener]) -> None:
        super().__init__(listeners=listeners)
        self._dirty: set[Key] = set()
        self._running: dict[Key, asyncio.Task[None]] = {}
        self._loose: set[asyncio.Task[None]] = set()

    async def __aenter__(self) -> ReconcilingEventEmitter:
        return self

    async def __aexit__(self, *_: object) -> None:
        pending = [*self._running.values(), *self._loose]
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    def emit(self, event_id: str, *args: Any, **kwargs: Any) -> None:
        for listener in self.listeners.get(event_id, ()):
            match self._key(listener, event_id, args, kwargs):
                case None:
                    self._spawn(listener, args, kwargs)
                case key:
                    self._dirty.add(key)
                    if key not in self._running:
                        self._running[key] = asyncio.create_task(self._drain(key, listener, args, kwargs))

    async def _drain(self, key: Key, listener: EventListener, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        try:
            while key in self._dirty:
                self._dirty.discard(key)
                await self._invoke(listener, args, kwargs)
        finally:
            del self._running[key]

    def _spawn(self, listener: EventListener, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        task = asyncio.create_task(self._invoke(listener, args, kwargs))
        self._loose.add(task)
        task.add_done_callback(self._loose.discard)

    async def _invoke(self, listener: EventListener, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        try:
            await listener.fn(*args, **kwargs)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("event listener failed: %s", listener.fn.__name__)

    @staticmethod
    def _key(listener: EventListener, event_id: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Key | None:
        try:
            return listener, event_id, hash((args, frozenset(kwargs.items())))
        except TypeError:
            return None
