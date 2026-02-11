"""Registry for distributed collections backed by Casty's native Distributed API."""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Callable
from typing import Any

from loguru import logger

from .proxies import (
    BarrierProxy,
    CounterProxy,
    DictProxy,
    LockProxy,
    QueueProxy,
    SetProxy,
)
from .types import Consistency

log = logger.bind(component="distributed")


def _call_on_loop[T](loop: asyncio.AbstractEventLoop, fn: Callable[[], T]) -> T:
    try:
        running = asyncio.get_running_loop()
        if running is loop:
            return fn()
    except RuntimeError:
        pass

    result_future: concurrent.futures.Future[T] = concurrent.futures.Future()

    def _run() -> None:
        try:
            result_future.set_result(fn())
        except Exception as e:
            result_future.set_exception(e)

    loop.call_soon_threadsafe(_run)
    return result_future.result(timeout=10)


class DistributedRegistry:
    __slots__ = ("_distributed", "_loop")

    def __init__(self, system: Any, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._loop = loop or asyncio.get_running_loop()
        self._distributed = system.distributed()

    def _get_distributed(self) -> Any:
        return self._distributed

    def dict(
        self,
        name: str,
        *,
        consistency: Consistency | None = None,
    ) -> DictProxy:
        log.debug("Creating distributed dict name={name}", name=name)
        d = self._get_distributed()
        map_ = _call_on_loop(self._loop, lambda: d.map[str, Any](name))
        return DictProxy(map_, consistency=consistency or "eventual")

    def set(
        self,
        name: str,
        *,
        consistency: Consistency | None = None,
    ) -> SetProxy:
        log.debug("Creating distributed set name={name}", name=name)
        d = self._get_distributed()
        set_ = _call_on_loop(self._loop, lambda: d.set[Any](name))
        return SetProxy(set_, consistency=consistency or "eventual")

    def counter(
        self,
        name: str,
        *,
        consistency: Consistency | None = None,
    ) -> CounterProxy:
        d = self._get_distributed()
        counter = _call_on_loop(self._loop, lambda: d.counter(name))
        return CounterProxy(counter, consistency=consistency or "eventual")

    def queue(self, name: str) -> QueueProxy:
        d = self._get_distributed()
        queue = _call_on_loop(self._loop, lambda: d.queue[Any](name))
        return QueueProxy(queue)

    def barrier(self, name: str, n: int) -> BarrierProxy:
        d = self._get_distributed()
        barrier = _call_on_loop(self._loop, lambda: d.barrier(name))
        return BarrierProxy(barrier, n)

    def lock(self, name: str) -> LockProxy:
        d = self._get_distributed()
        lock = _call_on_loop(self._loop, lambda: d.lock(name))
        return LockProxy(lock)

    def cleanup(self) -> None:
        log.debug("Cleaning up distributed registry")
        self._distributed = None
