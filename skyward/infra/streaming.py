"""Streaming helpers for @sky.function generator functions.

Generator functions stream results back through pull RPCs on the node's
``WorkerControl`` service (``next_chunk``); ``Iterator``-annotated params
are pushed element by element (``feed``/``feed_done``). This module holds
the casty-free pieces: generator detection, stream-param discovery, and
the synchronous iterator the client hands to user code.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any, get_type_hints

_SENTINEL = object()


class _SyncStream[T]:
    """Wraps an ``AsyncIterator[T]`` running on ``loop`` as a sync iterator.

    A single drain task consumes the async stream and feeds a thread-safe
    queue; the calling (user) thread reads from that queue.
    """

    def __init__(
        self,
        source: AsyncIterator[T],
        loop: asyncio.AbstractEventLoop,
        timeout: float = 300.0,
    ) -> None:
        import queue as _queue_mod

        self._q: _queue_mod.Queue[T | BaseException | object] = _queue_mod.Queue()
        self._empty = _queue_mod.Empty
        self._timeout = timeout
        self._task = loop.create_task(self._drain(source))

    async def _drain(self, source: AsyncIterator[T]) -> None:
        try:
            async for elem in source:
                self._q.put(elem)
        except BaseException as exc:
            self._q.put(exc)
        finally:
            self._q.put(_SENTINEL)

    def __iter__(self) -> _SyncStream[T]:
        return self

    def __next__(self) -> T:
        try:
            item = self._q.get(timeout=self._timeout)
        except self._empty:
            raise StopIteration from None
        if item is _SENTINEL:
            raise StopIteration
        match item:
            case BaseException():
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


async def iter_output_stream(
    control: Any,
    task_id: str,
    *,
    wait: float = 30.0,
) -> AsyncIterator[Any]:
    """Pull a task's generator output from ``WorkerControl.next_chunk``."""
    from skyward.infra.worker import STREAM_END, STREAM_ERROR, STREAM_ITEM, loads

    while True:
        kind, item = loads(await control.next_chunk(task_id, wait))
        match kind:
            case _ if kind == STREAM_ITEM:
                yield item
            case _ if kind == STREAM_END:
                return
            case _ if kind == STREAM_ERROR:
                raise RuntimeError(f"stream failed on task {task_id}: {item}")
            case _:  # pending — poll again
                continue
