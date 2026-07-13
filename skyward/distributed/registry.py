"""Registry for distributed collections backed by casty v2 collections.

Every collection is replicated across ``min(3, num_nodes)`` physical nodes
with quorum-acknowledged writes — a pool of one node runs with a single
replica so writes are never fenced by an unreachable quorum.
"""

from __future__ import annotations

import asyncio
from typing import Any

from skyward.observability.logger import logger

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


class DistributedRegistry:
    __slots__ = ("_loop", "_replicas", "_system")

    def __init__(
        self,
        system: Any,
        loop: asyncio.AbstractEventLoop | None = None,
        num_nodes: int = 1,
    ) -> None:
        self._loop = loop or asyncio.get_running_loop()
        self._system: Any = system
        self._replicas = min(3, max(1, num_nodes))

    def dict(
        self,
        name: str,
        *,
        consistency: Consistency | None = None,
    ) -> DictProxy:
        log.debug("Creating distributed dict name={name}", name=name)
        return DictProxy(
            self._system.map(name, replicas=self._replicas),
            consistency=consistency or "eventual",
        )

    def set(
        self,
        name: str,
        *,
        consistency: Consistency | None = None,
    ) -> SetProxy:
        log.debug("Creating distributed set name={name}", name=name)
        return SetProxy(
            self._system.set(name, replicas=self._replicas),
            consistency=consistency or "eventual",
        )

    def counter(
        self,
        name: str,
        *,
        consistency: Consistency | None = None,
    ) -> CounterProxy:
        log.debug("Creating distributed counter name={name}", name=name)
        return CounterProxy(
            self._system.counter(name, replicas=self._replicas),
            consistency=consistency or "eventual",
        )

    def queue(self, name: str) -> QueueProxy:
        log.debug("Creating distributed queue name={name}", name=name)
        return QueueProxy(self._system.queue(name, replicas=self._replicas))

    def barrier(self, name: str, n: int) -> BarrierProxy:
        log.debug("Creating distributed barrier name={name} n={n}", name=name, n=n)
        return BarrierProxy(
            self._system.barrier(name, parties=n, replicas=self._replicas), n,
        )

    def lock(self, name: str, timeout: float = 30) -> LockProxy:
        log.debug("Creating distributed lock name={name}", name=name)
        return LockProxy(
            self._system.lock(name, timeout=timeout, replicas=self._replicas),
            timeout,
        )

    def cleanup(self) -> None:
        log.debug("Cleaning up distributed registry")
        self._system = None
