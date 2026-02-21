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
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, get_type_hints

from casty import ActorRef, SourceRef


@dataclass(frozen=True, slots=True)
class _StreamHandle:
    producer_ref: ActorRef
    node_id: int


@dataclass(frozen=True, slots=True)
class _InputStreamRef:
    producer_ref: ActorRef


class _SyncSource[T]:
    """Wraps async SourceRef[T] as a synchronous iterator.

    Bridges the async Casty stream protocol to Python's __iter__/__next__
    for use in the synchronous ComputePool API.
    """

    def __init__(
        self,
        source: SourceRef[T],
        loop: asyncio.AbstractEventLoop,
        timeout: float = 300.0,
    ) -> None:
        self._source = source
        self._loop = loop
        self._timeout = timeout
        self._aiter: Any = None

    def __iter__(self) -> _SyncSource[T]:
        return self

    def __next__(self) -> T:
        aiter = self._aiter
        if aiter is None:
            aiter = self._source.__aiter__()  # type: ignore[assignment]
            self._aiter = aiter

        try:
            return asyncio.run_coroutine_threadsafe(
                aiter.__anext__(), self._loop,
            ).result(timeout=self._timeout)
        except StopAsyncIteration:
            raise StopIteration from None


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
