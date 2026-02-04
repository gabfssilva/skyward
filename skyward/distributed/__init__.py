"""Distributed collections for Skyward.

Public API:
    sky.dict("name")     - Distributed dict
    sky.list("name")     - Distributed list
    sky.set("name")      - Distributed set
    sky.counter("name")  - Distributed counter
    sky.queue("name")    - Distributed queue
    sky.barrier("name", n=N)  - Synchronization barrier
    sky.lock("name")     - Distributed lock
"""

from __future__ import annotations

from contextvars import ContextVar

from .types import Consistency
from .proxies import (
    CounterProxy,
    DictProxy,
    ListProxy,
    SetProxy,
    QueueProxy,
    BarrierProxy,
    LockProxy,
)
from .registry import DistributedRegistry


# Context variable for active registry
_active_registry: ContextVar[DistributedRegistry | None] = ContextVar(
    "active_registry", default=None
)


def _get_active_registry() -> DistributedRegistry:
    """Get the active registry.

    Works both locally (via ContextVar) and in remote Ray workers
    (by checking COMPUTE_POOL env var and connecting to existing actors).
    """
    import os

    reg = _active_registry.get()
    if reg is not None:
        return reg

    # Check if we're in a remote worker with COMPUTE_POOL set
    if os.environ.get("COMPUTE_POOL"):
        # Create a registry that connects to existing actors (get_if_exists=True)
        # This works because Ray actors are named and accessible cluster-wide
        reg = DistributedRegistry()
        _active_registry.set(reg)
        return reg

    raise RuntimeError(
        "No active pool. Use within a @pool decorated function or 'with pool():' block."
    )


def _set_active_registry(registry: DistributedRegistry | None) -> None:
    """Set the active registry (for internal use)."""
    _active_registry.set(registry)


# Public API functions


def dict(name: str, *, consistency: Consistency | None = None) -> DictProxy:
    """Get or create a distributed dict.

    Args:
        name: Unique name for the dict.
        consistency: "strong" (wait for writes) or "eventual" (fire-and-forget).

    Returns:
        DictProxy with dict-like interface.

    Example:
        cache = sky.dict("embeddings")
        cache["key"] = value
        v = cache["key"]
    """
    return _get_active_registry().dict(name, consistency=consistency)


def list(name: str, *, consistency: Consistency | None = None) -> ListProxy:
    """Get or create a distributed list.

    Args:
        name: Unique name for the list.
        consistency: "strong" (wait for writes) or "eventual" (fire-and-forget).

    Returns:
        ListProxy with list-like interface.

    Example:
        results = sky.list("outputs")
        results.append(value)
    """
    return _get_active_registry().list(name, consistency=consistency)


def set(name: str, *, consistency: Consistency | None = None) -> SetProxy:
    """Get or create a distributed set.

    Args:
        name: Unique name for the set.
        consistency: "strong" (wait for writes) or "eventual" (fire-and-forget).

    Returns:
        SetProxy with set-like interface.

    Example:
        seen = sky.set("processed")
        seen.add(item)
        if item in seen: ...
    """
    return _get_active_registry().set(name, consistency=consistency)


def counter(name: str, *, consistency: Consistency | None = None) -> CounterProxy:
    """Get or create a distributed counter.

    Args:
        name: Unique name for the counter.
        consistency: "strong" (wait for writes) or "eventual" (fire-and-forget).

    Returns:
        CounterProxy with counter interface.

    Example:
        progress = sky.counter("steps")
        progress.increment()
        print(progress.value)
    """
    return _get_active_registry().counter(name, consistency=consistency)


def queue(name: str) -> QueueProxy:
    """Get or create a distributed queue.

    Note: Queue always uses strong consistency for FIFO semantics.

    Args:
        name: Unique name for the queue.

    Returns:
        QueueProxy with queue interface.

    Example:
        tasks = sky.queue("work")
        tasks.put(item)
        item = tasks.get()
    """
    return _get_active_registry().queue(name)


def barrier(name: str, n: int) -> BarrierProxy:
    """Get or create a distributed barrier.

    Note: Barrier always uses strong consistency.

    Args:
        name: Unique name for the barrier.
        n: Number of parties that must arrive before release.

    Returns:
        BarrierProxy with barrier interface.

    Example:
        sync = sky.barrier("epoch", n=4)
        sync.wait()  # Blocks until 4 arrive
    """
    return _get_active_registry().barrier(name, n)


def lock(name: str) -> LockProxy:
    """Get or create a distributed lock.

    Note: Lock always uses strong consistency.

    Args:
        name: Unique name for the lock.

    Returns:
        LockProxy with lock interface.

    Example:
        lock = sky.lock("critical")
        with lock:
            # Critical section
    """
    return _get_active_registry().lock(name)


__all__ = [
    # Types
    "Consistency",
    # Proxies
    "CounterProxy",
    "DictProxy",
    "ListProxy",
    "SetProxy",
    "QueueProxy",
    "BarrierProxy",
    "LockProxy",
    # Registry
    "DistributedRegistry",
    # Functions
    "dict",
    "list",
    "set",
    "counter",
    "queue",
    "barrier",
    "lock",
    # Internal
    "_get_active_registry",
    "_set_active_registry",
]
