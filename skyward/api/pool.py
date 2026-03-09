"""Public Pool protocol — the interface returned by Compute and Session.

Defines the structural contract that every pool implementation must
satisfy.  Consumer code depends on this protocol, never on the concrete
``ComputePool`` class.  All task-dispatch operators (``>>``, ``@``, ``>``)
ultimately delegate to methods defined here.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from skyward.api.distributed import Consistency
    from skyward.api.function import PendingFunction, PendingFunctionGroup
    from skyward.distributed import (
        BarrierProxy,
        CounterProxy,
        DictProxy,
        LockProxy,
        QueueProxy,
        SetProxy,
    )


@runtime_checkable
class Pool(Protocol):
    """A provisioned pool of cloud compute nodes.

    Returned by ``sky.Compute()`` and ``session.compute()``.
    Dispatch tasks via operators or the equivalent methods below.

    Examples
    --------
    >>> with sky.Compute(provider=sky.AWS(), accelerator="A100", nodes=4) as pool:
    ...     result = train(data) >> pool        # one node (round-robin)
    ...     results = train(data) @ pool        # broadcast to ALL nodes
    ...     a, b = (task1() & task2()) >> pool  # parallel execution
    ...     future = train(data) > pool         # async, returns Future[T]
    """

    @property
    def concurrency(self) -> int:
        """Number of concurrent task slots available per node.

        Determined by ``Worker.concurrency`` in the pool options.
        A pool with 4 nodes and ``concurrency=2`` can run 8 tasks
        simultaneously.

        Returns
        -------
        int
            Slots per node.
        """
        ...

    @property
    def is_active(self) -> bool:
        """Whether the pool is provisioned and ready for task dispatch.

        Returns ``False`` before ``__enter__`` completes and after
        ``__exit__`` begins teardown.

        Returns
        -------
        bool
            ``True`` when the pool can accept tasks.
        """
        ...

    def run[T](self, pending: PendingFunction[T]) -> T:
        """Execute a pending function on one node (round-robin).

        This is the method behind the ``task() >> pool`` operator.
        Blocks until the remote function returns.

        Parameters
        ----------
        pending
            A ``PendingFunction`` produced by calling a ``@sky.function``.

        Returns
        -------
        T
            The remote function's return value, deserialized.

        Raises
        ------
        RuntimeError
            If the pool is not active or the remote execution fails.
        """
        ...

    def run_async[T](self, pending: PendingFunction[T]) -> Future[T]:
        """Submit a pending function for non-blocking execution.

        This is the method behind the ``task() > pool`` operator.
        Returns immediately with a ``Future`` that resolves when the
        remote function completes.

        Parameters
        ----------
        pending
            A ``PendingFunction`` produced by calling a ``@sky.function``.

        Returns
        -------
        Future[T]
            A future that resolves to the remote return value.
        """
        ...

    def broadcast[T](self, pending: PendingFunction[T]) -> list[T]:
        """Execute a pending function on every node in the pool.

        This is the method behind the ``task() @ pool`` operator.
        Blocks until all nodes return. Results are ordered by node index.

        Parameters
        ----------
        pending
            A ``PendingFunction`` produced by calling a ``@sky.function``.

        Returns
        -------
        list[T]
            One result per node, ordered by node index.
        """
        ...

    def run_parallel(
        self, group: PendingFunctionGroup,
    ) -> tuple[Any, ...] | Generator[Any, None, None]:
        """Execute a group of pending functions concurrently.

        This is the method behind the ``(a() & b()) >> pool`` operator.
        Tasks are distributed across available nodes. Blocks until all
        tasks complete.

        Parameters
        ----------
        group
            A ``PendingFunctionGroup`` created via ``&`` or ``gather()``.

        Returns
        -------
        tuple[Any, ...] | Generator[Any, None, None]
            Results from all tasks. A tuple when ``stream=False``
            (default), or a generator when ``stream=True``.
        """
        ...

    def map[T, R](
        self, fn: Callable[[T], R], items: Sequence[T],
    ) -> list[R]:
        """Apply a function to each item, distributing across nodes.

        Wraps *fn* in ``@sky.function`` internally and dispatches one
        task per item using round-robin scheduling. Results are returned
        in the same order as *items*.

        Parameters
        ----------
        fn
            A callable that takes one item and returns a result.
        items
            Sequence of inputs to map over.

        Returns
        -------
        list[R]
            Results in the same order as *items*.
        """
        ...

    def current_nodes(self) -> int:
        """Return the number of nodes currently ready for task dispatch.

        For autoscaling pools, this may change between calls as nodes
        are provisioned or drained.

        Returns
        -------
        int
            Count of ready nodes.
        """
        ...

    def dict(self, name: str, *, consistency: Consistency | None = None) -> DictProxy:
        """Get or create a named distributed dictionary.

        Distributed dictionaries are shared across all nodes in the pool.
        Operations are forwarded to the head node's actor system.

        Parameters
        ----------
        name
            Unique name for the dictionary within this pool.
        consistency
            ``"strong"`` (linearizable, default) or ``"eventual"``.

        Returns
        -------
        DictProxy
            A dict-like proxy that synchronizes across nodes.
        """
        ...

    def set(self, name: str, *, consistency: Consistency | None = None) -> SetProxy:
        """Get or create a named distributed set.

        Parameters
        ----------
        name
            Unique name for the set within this pool.
        consistency
            ``"strong"`` (linearizable, default) or ``"eventual"``.

        Returns
        -------
        SetProxy
            A set-like proxy that synchronizes across nodes.
        """
        ...

    def counter(self, name: str, *, consistency: Consistency | None = None) -> CounterProxy:
        """Get or create a named distributed counter.

        Parameters
        ----------
        name
            Unique name for the counter within this pool.
        consistency
            ``"strong"`` (linearizable, default) or ``"eventual"``.

        Returns
        -------
        CounterProxy
            An atomic counter proxy that synchronizes across nodes.
        """
        ...

    def queue(self, name: str) -> QueueProxy:
        """Get or create a named distributed FIFO queue.

        Parameters
        ----------
        name
            Unique name for the queue within this pool.

        Returns
        -------
        QueueProxy
            A queue proxy that synchronizes across nodes.
        """
        ...

    def barrier(self, name: str, n: int) -> BarrierProxy:
        """Get or create a named distributed barrier.

        All *n* participants must call ``wait()`` before any of them
        proceed.  Useful for synchronizing phases across nodes.

        Parameters
        ----------
        name
            Unique name for the barrier within this pool.
        n
            Number of participants that must arrive before releasing.

        Returns
        -------
        BarrierProxy
            A barrier proxy that synchronizes across nodes.
        """
        ...

    def lock(self, name: str, timeout: float = 30) -> LockProxy:
        """Get or create a named distributed lock.

        Provides mutual exclusion across nodes.  Supports the context
        manager protocol (``with pool.lock("name"):``).

        Parameters
        ----------
        name
            Unique name for the lock within this pool.
        timeout
            Maximum seconds to wait for lock acquisition.

        Returns
        -------
        LockProxy
            A lock proxy that synchronizes across nodes.
        """
        ...
