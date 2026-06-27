"""Per-pool in-memory execution-event history for ``sky log`` export.

A bounded ring buffer fed by one projection ``on_event`` tap — the same
seam SSE uses. It stores already-serialized ``event_to_json`` dicts so the
log route is a thin reader, plus a per-pool ``(eid → source)`` map (fed by
the ``X-Skyward-Source`` header on script executions) so ipynb code cells
carry the executed source. Volatile (lost on restart) and intentionally
survives ``delete_pool`` so post-mortem export still works.
"""

from __future__ import annotations

from collections import deque
from threading import Lock
from typing import TYPE_CHECKING

from skyward.server.wire import event_to_json

if TYPE_CHECKING:
    from collections.abc import Callable

    from skyward.api.events import SessionEvent
    from skyward.api.projection import SessionProjection

DEFAULT_CAPACITY = 10_000


class HistoryStore:
    """Thread-safe bounded ring of serialized events, keyed by pool name."""

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        self._capacity = capacity
        self._lock = Lock()
        self._by_pool: dict[str, deque[dict]] = {}
        self._sources: dict[str, dict[str, str]] = {}

    def append(self, event: SessionEvent) -> None:
        """Record an event under its ``pool_name`` (ignored if absent)."""
        name = getattr(event, "pool_name", None)
        if not isinstance(name, str):
            return
        payload = event_to_json(event)
        with self._lock:
            buf = self._by_pool.get(name)
            if buf is None:
                buf = deque(maxlen=self._capacity)
                self._by_pool[name] = buf
            buf.append(payload)

    def set_source(self, name: str, eid: str, source: str) -> None:
        """Associate an execution id with its submitted script source."""
        with self._lock:
            self._sources.setdefault(name, {})[eid] = source

    def get(self, name: str, limit: int | None = None) -> list[dict]:
        """Return recorded events for a pool, optionally the last *limit*."""
        with self._lock:
            buf = self._by_pool.get(name)
            items = list(buf) if buf is not None else []
        return items[-limit:] if limit else items

    def sources(self, name: str) -> dict[str, str]:
        """Return the ``{eid: source}`` map for a pool."""
        with self._lock:
            return dict(self._sources.get(name, {}))

    def has(self, name: str) -> bool:
        """Whether any history or source exists for a pool."""
        with self._lock:
            return name in self._by_pool or name in self._sources

    def drop(self, name: str) -> None:
        """Forget all history and sources for a pool."""
        with self._lock:
            self._by_pool.pop(name, None)
            self._sources.pop(name, None)


def attach_history(projection: SessionProjection, store: HistoryStore) -> Callable[[], None]:
    """Subscribe ``store.append`` to the projection's event stream.

    Returns the idempotent unsubscribe handle.
    """
    return projection.subscribe(on_event=store.append)
