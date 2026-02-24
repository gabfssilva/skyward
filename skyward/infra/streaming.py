"""Streaming primitives for @sky.compute generator functions.

Supports bidirectional streaming:
- Output: generator functions stream results back as a sync iterator
- Input: Iterator[T]-annotated params stream data to workers incrementally

Uses Casty's stream_producer/stream_consumer actors for backpressure-aware
streaming over SSH-tunneled TCP connections.
"""

from __future__ import annotations

import asyncio
import inspect
import queue as _queue_mod
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import get_type_hints

from casty import ActorRef, SourceRef

_SENTINEL = object()


@dataclass(frozen=True, slots=True)
class _StreamHandle:
    producer_ref: ActorRef
    node_id: int


@dataclass(frozen=True, slots=True)
class _InputStreamRef:
    producer_ref: ActorRef


class _SyncSource[T]:
    """Wraps async SourceRef[T] as a synchronous iterator.

    Runs a single drain task on the event loop that consumes the full
    async stream and feeds elements into a thread-safe queue.  The
    calling (main) thread reads from that queue — no cross-thread
    async-generator interaction.
    """

    def __init__(
        self,
        source: SourceRef[T],
        loop: asyncio.AbstractEventLoop,
        timeout: float = 300.0,
    ) -> None:
        self._q: _queue_mod.Queue[T | BaseException | object] = _queue_mod.Queue()
        self._timeout = timeout
        self._task = loop.create_task(self._drain(source))

    async def _drain(self, source: SourceRef[T]) -> None:
        try:
            async for elem in source:
                self._q.put(elem)
        except BaseException as exc:
            self._q.put(exc)
        finally:
            self._q.put(_SENTINEL)

    def __iter__(self) -> _SyncSource[T]:
        return self

    def __next__(self) -> T:
        try:
            item = self._q.get(timeout=self._timeout)
        except _queue_mod.Empty:
            raise StopIteration from None
        if item is _SENTINEL:
            raise StopIteration
        if isinstance(item, BaseException):
            raise item
        return item  # type: ignore[return-value]


def _unwrap(fn: Callable) -> Callable:  # type: ignore[type-arg]
    return inspect.unwrap(fn)


def is_generator_compute(fn: Callable) -> bool:  # type: ignore[type-arg]
    return inspect.isgeneratorfunction(_unwrap(fn))


def _is_iterator_hint(hint: type) -> bool:
    origin = getattr(hint, "__origin__", None)
    return origin is Iterator or origin is iter


def _stream_param_indices(fn: Callable) -> tuple[int, ...]:  # type: ignore[type-arg]
    unwrapped = _unwrap(fn)

    try:
        hints = get_type_hints(unwrapped)
    except Exception:
        return ()

    params = list(inspect.signature(unwrapped).parameters.values())
    return tuple(
        i for i, p in enumerate(params)
        if p.name in hints and _is_iterator_hint(hints[p.name])
    )
