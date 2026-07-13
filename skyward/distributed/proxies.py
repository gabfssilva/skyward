"""Proxy wrappers for casty v2's distributed collections.

Casty collections are async-only. These proxies provide synchronous
access by submitting coroutines to the system's event loop from any thread.

Values are cloudpickled to ``bytes`` before hitting casty's msgpack wire,
so arbitrary Python objects (including ``None``) round-trip unchanged.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Coroutine
from typing import Any

import cloudpickle

from skyward.observability.logger import logger

from .types import Consistency

log = logger.bind(component="distributed")

_system_loop: asyncio.AbstractEventLoop | None = None


def set_system_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _system_loop
    _system_loop = loop
    log.debug("System event loop registered for distributed proxies")


def _get_loop() -> asyncio.AbstractEventLoop:
    if _system_loop is None:
        raise RuntimeError("No system event loop set for distributed collections")
    return _system_loop


def _run_sync[T](coro: Coroutine[Any, Any, T], *, timeout: float = 30) -> T:
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
    try:
        return future.result(timeout=timeout)
    except TimeoutError:
        log.warning("Cross-thread coroutine submission timed out after {t}s", t=timeout)
        raise
    except Exception:
        log.debug("Cross-thread coroutine submission failed")
        raise


def _enc(value: Any) -> bytes:
    return cloudpickle.dumps(value)


def _dec(data: bytes | None) -> Any:
    return None if data is None else cloudpickle.loads(data)


class CounterProxy:
    """Synchronous proxy for a distributed counter.

    Thread-safe — submits operations to the actor system's event loop.
    Use inside ``@sky.function`` or on the client via ``pool.counter()``.

    Examples
    --------
    >>> counter = sky.counter("processed")
    >>> counter.increment()
    >>> counter.increment(5)
    >>> print(counter.value)  # 6
    >>> int(counter)          # 6
    """

    __slots__ = ("_consistency", "_counter")

    def __init__(self, counter: Any, consistency: Consistency = "eventual") -> None:
        self._counter = counter
        self._consistency = consistency

    @property
    def value(self) -> int:
        return _run_sync(self._counter.get())

    def increment(self, n: int = 1) -> None:
        _run_sync(self._counter.add(n))

    def decrement(self, n: int = 1) -> None:
        _run_sync(self._counter.add(-n))

    def reset(self, value: int = 0) -> None:
        _run_sync(self._reset_to(value))

    def __int__(self) -> int:
        return self.value

    async def _reset_to(self, value: int) -> None:
        await self._counter.reset()
        if value:
            await self._counter.add(value)

    async def value_async(self) -> int:
        return await self._counter.get()

    async def increment_async(self, n: int = 1) -> None:
        await self._counter.add(n)

    async def decrement_async(self, n: int = 1) -> None:
        await self._counter.add(-n)

    async def reset_async(self, value: int = 0) -> None:
        await self._reset_to(value)


class DictProxy:
    """Synchronous proxy for a distributed dictionary.

    Support standard dict operations (``[]``, ``in``, ``get``, ``pop``).
    Thread-safe — submits operations to the actor system's event loop.

    Examples
    --------
    >>> metrics = sky.dict("metrics")
    >>> metrics["loss"] = 0.5
    >>> metrics["lr"] = 1e-3
    >>> print(metrics["loss"])      # 0.5
    >>> print("loss" in metrics)    # True
    >>> metrics.update({"a": 1, "b": 2})
    """

    __slots__ = ("_consistency", "_map")

    def __init__(self, map_: Any, consistency: Consistency = "eventual") -> None:
        self._map = map_
        self._consistency = consistency

    def __getitem__(self, key: str) -> Any:
        result = _run_sync(self._map.get(key))
        if result is None:
            raise KeyError(key)
        return _dec(result)

    def __setitem__(self, key: str, value: Any) -> None:
        _run_sync(self._map.put(key, _enc(value)))

    def __delitem__(self, key: str) -> None:
        _run_sync(self._map.remove(key))

    def __contains__(self, key: str) -> bool:
        return _run_sync(self._map.contains(key))

    def get(self, key: str, default: Any = None) -> Any:
        result = _run_sync(self._map.get(key))
        return _dec(result) if result is not None else default

    def update(self, items: dict[str, Any]) -> None:
        for k, v in items.items():
            _run_sync(self._map.put(k, _enc(v)))

    def pop(self, key: str, default: Any = None) -> Any:
        result = _run_sync(self._map.get(key))
        if result is not None:
            _run_sync(self._map.remove(key))
            return _dec(result)
        return default

    async def get_async(self, key: str, default: Any = None) -> Any:
        result = await self._map.get(key)
        return _dec(result) if result is not None else default

    async def set_async(self, key: str, value: Any) -> None:
        await self._map.put(key, _enc(value))

    async def update_async(self, items: dict[str, Any]) -> None:
        for k, v in items.items():
            await self._map.put(k, _enc(v))

    async def pop_async(self, key: str, default: Any = None) -> Any:
        result = await self._map.get(key)
        if result is not None:
            await self._map.remove(key)
            return _dec(result)
        return default


class SetProxy:
    """Synchronous proxy for a distributed set.

    Thread-safe — submits operations to the actor system's event loop.

    Examples
    --------
    >>> seen = sky.set("seen_ids")
    >>> seen.add("abc")
    >>> print("abc" in seen)  # True
    >>> seen.discard("abc")
    """

    __slots__ = ("_consistency", "_set")

    def __init__(self, set_: Any, consistency: Consistency = "eventual") -> None:
        self._set = set_
        self._consistency = consistency

    def __contains__(self, value: Any) -> bool:
        return _run_sync(self._set.contains(_enc(value)))

    def __len__(self) -> int:
        return _run_sync(self._set.size())

    def add(self, value: Any) -> None:
        _run_sync(self._set.add(_enc(value)))

    def discard(self, value: Any) -> None:
        _run_sync(self._set.remove(_enc(value)))

    async def add_async(self, value: Any) -> None:
        await self._set.add(_enc(value))

    async def discard_async(self, value: Any) -> None:
        await self._set.remove(_enc(value))

    async def contains_async(self, value: Any) -> bool:
        return await self._set.contains(_enc(value))


class QueueProxy:
    """Synchronous proxy for a distributed FIFO queue.

    Thread-safe — submits operations to the actor system's event loop.

    Examples
    --------
    >>> q = sky.queue("tasks")
    >>> q.put("item1")
    >>> q.put("item2")
    >>> print(q.get())           # "item1"
    >>> print(q.get(timeout=5))  # "item2" (waits up to 5s)
    """

    __slots__ = ("_queue",)

    def __init__(self, queue: Any) -> None:
        self._queue = queue

    def __len__(self) -> int:
        return _run_sync(self._queue.size())

    def put(self, value: Any) -> None:
        _run_sync(self._queue.offer(_enc(value)))

    def get(self, timeout: float | None = None) -> Any:
        start = time.monotonic()
        delay = 0.01
        while True:
            result = _run_sync(self._queue.poll())
            if result is not None:
                return _dec(result)
            if timeout is not None and time.monotonic() - start >= timeout:
                return None
            time.sleep(delay)
            delay = min(delay * 1.5, 0.5)

    def empty(self) -> bool:
        return _run_sync(self._queue.size()) == 0

    async def put_async(self, value: Any) -> None:
        await self._queue.offer(_enc(value))

    async def get_async(self, timeout: float | None = None) -> Any:
        start = time.monotonic()
        delay = 0.01
        while True:
            result = await self._queue.poll()
            if result is not None:
                return _dec(result)
            if timeout is not None and time.monotonic() - start >= timeout:
                return None
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 0.5)


class BarrierProxy:
    """Synchronous proxy for a distributed barrier.

    All participating nodes call ``wait()`` and block until all have arrived.

    Examples
    --------
    >>> barrier = sky.barrier("sync_point", n=4)
    >>> # ... each node does work ...
    >>> barrier.wait()  # blocks until all 4 arrive
    """

    __slots__ = ("_barrier", "_n")

    def __init__(self, barrier: Any, n: int) -> None:
        self._barrier = barrier
        self._n = n

    def wait(self) -> None:
        _run_sync(self._barrier.wait(), timeout=86400.0)

    def reset(self) -> None:
        """No-op: the casty v2 barrier is cyclic — completing a generation
        resets it for the next round automatically."""

    async def wait_async(self) -> None:
        await self._barrier.wait()


class LockProxy:
    """Synchronous proxy for a distributed lock.

    Support both explicit ``acquire``/``release`` and context manager usage.
    The underlying lock is a TTL lease; the proxy renews it in the background
    while held, so long critical sections don't lose the lock.

    Examples
    --------
    >>> lock = sky.lock("critical_section")
    >>> with lock:
    ...     update_shared_state()

    >>> # Or explicitly
    >>> lock.acquire()
    >>> try:
    ...     update_shared_state()
    ... finally:
    ...     lock.release()
    """

    _TTL = 60.0

    __slots__ = ("_lease", "_lock", "_renewer", "_timeout")

    def __init__(self, lock: Any, timeout: float) -> None:
        self._lock = lock
        self._timeout = timeout
        self._lease: Any = None
        self._renewer: Any = None

    async def _renew_loop(self, lease: Any) -> None:
        while True:
            await asyncio.sleep(self._TTL / 2)
            if not await lease.renew(self._TTL):
                return

    async def _acquire(self) -> None:
        self._lease = await self._lock.acquire(ttl=self._TTL, timeout=self._timeout)
        self._renewer = asyncio.get_running_loop().create_task(
            self._renew_loop(self._lease),
        )

    async def _release(self) -> None:
        if self._renewer is not None:
            self._renewer.cancel()
            self._renewer = None
        if self._lease is not None:
            lease, self._lease = self._lease, None
            await lease.release()

    def acquire(self) -> bool:
        _run_sync(self._acquire(), timeout=self._timeout + 5.0)
        return True

    def release(self) -> None:
        _run_sync(self._release())

    def __enter__(self) -> LockProxy:
        self.acquire()
        return self

    def __exit__(self, *args: object) -> None:
        self.release()

    async def acquire_async(self) -> bool:
        await self._acquire()
        return True

    async def release_async(self) -> None:
        await self._release()

    async def __aenter__(self) -> LockProxy:
        await self.acquire_async()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.release_async()
