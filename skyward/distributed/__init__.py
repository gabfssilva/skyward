"""Distributed collections for Skyward.

Public API:
    sky.dict("name")     - Distributed dict
    sky.set("name")      - Distributed set
    sky.counter("name")  - Distributed counter
    sky.queue("name")    - Distributed queue
    sky.barrier("name", n=N)  - Synchronization barrier
    sky.lock("name")     - Distributed lock
"""

from __future__ import annotations

from contextvars import ContextVar

from .proxies import (
    BarrierProxy,
    CounterProxy,
    DictProxy,
    LockProxy,
    QueueProxy,
    SetProxy,
)
from .registry import DistributedRegistry
from .types import Consistency, Registry

_active_registry: ContextVar[Registry | None] = ContextVar(
    "active_registry", default=None
)


def _get_active_registry() -> Registry:
    reg = _active_registry.get()
    if reg is not None:
        return reg

    raise RuntimeError(
        "No active pool. Use within a @pool decorated function or 'with pool():' block."
    )


def _set_active_registry(registry: Registry | None) -> None:
    _active_registry.set(registry)


def dict(name: str, *, consistency: Consistency | None = None) -> DictProxy:
    return _get_active_registry().dict(name, consistency=consistency)


def set(name: str, *, consistency: Consistency | None = None) -> SetProxy:
    return _get_active_registry().set(name, consistency=consistency)


def counter(name: str, *, consistency: Consistency | None = None) -> CounterProxy:
    return _get_active_registry().counter(name, consistency=consistency)


def queue(name: str) -> QueueProxy:
    return _get_active_registry().queue(name)


def barrier(name: str, n: int) -> BarrierProxy:
    return _get_active_registry().barrier(name, n)


def lock(name: str) -> LockProxy:
    return _get_active_registry().lock(name)


__all__ = [
    "Consistency",
    "CounterProxy",
    "DictProxy",
    "SetProxy",
    "QueueProxy",
    "BarrierProxy",
    "LockProxy",
    "DistributedRegistry",
    "Registry",
    "dict",
    "set",
    "counter",
    "queue",
    "barrier",
    "lock",
    "_get_active_registry",
    "_set_active_registry",
]
