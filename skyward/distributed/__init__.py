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
    """Get or create a distributed dictionary (runtime shortcut).

    Must be called from within a ``@sky.function`` running on a pool.

    Parameters
    ----------
    name
        Unique identifier for this collection.
    consistency
        Consistency level. ``None`` uses the system default.

    Returns
    -------
    DictProxy
        Synchronous dict-like proxy.

    Examples
    --------
    >>> metrics = sky.dict("metrics")
    >>> metrics["loss"] = 0.5
    """
    return _get_active_registry().dict(name, consistency=consistency)


def set(name: str, *, consistency: Consistency | None = None) -> SetProxy:
    """Get or create a distributed set (runtime shortcut).

    Must be called from within a ``@sky.function`` running on a pool.

    Parameters
    ----------
    name
        Unique identifier for this collection.
    consistency
        Consistency level. ``None`` uses the system default.

    Returns
    -------
    SetProxy
        Synchronous set-like proxy.
    """
    return _get_active_registry().set(name, consistency=consistency)


def counter(name: str, *, consistency: Consistency | None = None) -> CounterProxy:
    """Get or create a distributed counter (runtime shortcut).

    Must be called from within a ``@sky.function`` running on a pool.

    Parameters
    ----------
    name
        Unique identifier for this collection.
    consistency
        Consistency level. ``None`` uses the system default.

    Returns
    -------
    CounterProxy
        Synchronous counter proxy.
    """
    return _get_active_registry().counter(name, consistency=consistency)


def queue(name: str) -> QueueProxy:
    """Get or create a distributed FIFO queue (runtime shortcut).

    Must be called from within a ``@sky.function`` running on a pool.

    Parameters
    ----------
    name
        Unique identifier for this collection.

    Returns
    -------
    QueueProxy
        Synchronous queue proxy.
    """
    return _get_active_registry().queue(name)


def barrier(name: str, n: int) -> BarrierProxy:
    """Get or create a distributed barrier (runtime shortcut).

    Must be called from within a ``@sky.function`` running on a pool.

    Parameters
    ----------
    name
        Unique identifier for this barrier.
    n
        Number of participants that must arrive before releasing.

    Returns
    -------
    BarrierProxy
        Synchronous barrier proxy.
    """
    return _get_active_registry().barrier(name, n)


def lock(name: str, timeout: float = 30) -> LockProxy:
    """Get or create a distributed lock (runtime shortcut).

    Must be called from within a ``@sky.function`` running on a pool.

    Parameters
    ----------
    name
        Unique identifier for this lock.
    timeout
        Maximum seconds to wait for acquisition. Default ``30``.

    Returns
    -------
    LockProxy
        Synchronous lock proxy (supports context manager).
    """
    return _get_active_registry().lock(name, timeout)


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
