"""Proxy wrappers for Casty's native distributed collections.

Casty collections are async-only. These proxies provide synchronous
access by submitting coroutines to the system's event loop from any thread.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from .types import Consistency

_system_loop: asyncio.AbstractEventLoop | None = None


def set_system_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _system_loop
    _system_loop = loop


def _get_loop() -> asyncio.AbstractEventLoop:
    if _system_loop is None:
        raise RuntimeError("No system event loop set for distributed collections")
    return _system_loop


def _run_sync[T](coro: Any) -> T:
    loop = _get_loop()

    try:
        running = asyncio.get_running_loop()
        if running is loop:
            raise RuntimeError(
                "Cannot call sync proxy from the system event loop; use async methods"
            )
    except RuntimeError as e:
        if "Cannot call sync" in str(e):
            raise

    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=30)


class CounterProxy:
    __slots__ = ("_counter", "_consistency")

    def __init__(self, counter: Any, consistency: Consistency = "eventual") -> None:
        self._counter = counter
        self._consistency = consistency

    @property
    def value(self) -> int:
        return _run_sync(self._counter.get())

    def increment(self, n: int = 1) -> None:
        _run_sync(self._counter.increment(n))

    def decrement(self, n: int = 1) -> None:
        _run_sync(self._counter.decrement(n))

    def reset(self, value: int = 0) -> None:
        _run_sync(self._counter.increment(value - _run_sync(self._counter.get())))

    def __int__(self) -> int:
        return self.value

    async def value_async(self) -> int:
        return await self._counter.get()

    async def increment_async(self, n: int = 1) -> None:
        await self._counter.increment(n)

    async def decrement_async(self, n: int = 1) -> None:
        await self._counter.decrement(n)

    async def reset_async(self, value: int = 0) -> None:
        current = await self._counter.get()
        await self._counter.increment(value - current)


class DictProxy:
    __slots__ = ("_map", "_consistency")

    def __init__(self, map_: Any, consistency: Consistency = "eventual") -> None:
        self._map = map_
        self._consistency = consistency

    def __getitem__(self, key: str) -> Any:
        result = _run_sync(self._map.get(key))
        if result is None:
            raise KeyError(key)
        return result

    def __setitem__(self, key: str, value: Any) -> None:
        _run_sync(self._map.put(key, value))

    def __delitem__(self, key: str) -> None:
        _run_sync(self._map.delete(key))

    def __contains__(self, key: str) -> bool:
        return _run_sync(self._map.contains(key))

    def get(self, key: str, default: Any = None) -> Any:
        result = _run_sync(self._map.get(key))
        return result if result is not None else default

    def update(self, items: dict[str, Any]) -> None:
        for k, v in items.items():
            _run_sync(self._map.put(k, v))

    def pop(self, key: str, default: Any = None) -> Any:
        result = _run_sync(self._map.get(key))
        if result is not None:
            _run_sync(self._map.delete(key))
            return result
        return default

    async def get_async(self, key: str, default: Any = None) -> Any:
        result = await self._map.get(key)
        return result if result is not None else default

    async def set_async(self, key: str, value: Any) -> None:
        await self._map.put(key, value)

    async def update_async(self, items: dict[str, Any]) -> None:
        for k, v in items.items():
            await self._map.put(k, v)

    async def pop_async(self, key: str, default: Any = None) -> Any:
        result = await self._map.get(key)
        if result is not None:
            await self._map.delete(key)
            return result
        return default


class SetProxy:
    __slots__ = ("_set", "_consistency")

    def __init__(self, set_: Any, consistency: Consistency = "eventual") -> None:
        self._set = set_
        self._consistency = consistency

    def __contains__(self, value: Any) -> bool:
        return _run_sync(self._set.contains(value))

    def __len__(self) -> int:
        return _run_sync(self._set.size())

    def add(self, value: Any) -> None:
        _run_sync(self._set.add(value))

    def discard(self, value: Any) -> None:
        _run_sync(self._set.remove(value))

    async def add_async(self, value: Any) -> None:
        await self._set.add(value)

    async def discard_async(self, value: Any) -> None:
        await self._set.remove(value)

    async def contains_async(self, value: Any) -> bool:
        return await self._set.contains(value)


class QueueProxy:
    __slots__ = ("_queue",)

    def __init__(self, queue: Any) -> None:
        self._queue = queue

    def __len__(self) -> int:
        return _run_sync(self._queue.size())

    def put(self, value: Any) -> None:
        _run_sync(self._queue.enqueue(value))

    def get(self, timeout: float | None = None) -> Any:
        start = time.monotonic()
        while True:
            result = _run_sync(self._queue.dequeue())
            if result is not None:
                return result
            if timeout is not None:
                elapsed = time.monotonic() - start
                if elapsed >= timeout:
                    return None
            time.sleep(0.01)

    def empty(self) -> bool:
        return _run_sync(self._queue.size()) == 0

    async def put_async(self, value: Any) -> None:
        await self._queue.enqueue(value)

    async def get_async(self, timeout: float | None = None) -> Any:
        start = time.monotonic()
        while True:
            result = await self._queue.dequeue()
            if result is not None:
                return result
            if timeout is not None:
                elapsed = time.monotonic() - start
                if elapsed >= timeout:
                    return None
            await asyncio.sleep(0.01)


class BarrierProxy:
    __slots__ = ("_barrier", "_n")

    def __init__(self, barrier: Any, n: int) -> None:
        self._barrier = barrier
        self._n = n

    def wait(self) -> None:
        _run_sync(self._barrier.arrive(self._n))

    def reset(self) -> None:
        pass

    async def wait_async(self) -> None:
        await self._barrier.arrive(self._n)


class LockProxy:
    __slots__ = ("_lock",)

    def __init__(self, lock: Any) -> None:
        self._lock = lock

    def acquire(self) -> bool:
        _run_sync(self._lock.acquire())
        return True

    def release(self) -> None:
        _run_sync(self._lock.release())

    def __enter__(self) -> LockProxy:
        self.acquire()
        return self

    def __exit__(self, *args: object) -> None:
        self.release()

    async def acquire_async(self) -> bool:
        await self._lock.acquire()
        return True

    async def release_async(self) -> None:
        await self._lock.release()

    async def __aenter__(self) -> LockProxy:
        await self.acquire_async()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.release_async()
