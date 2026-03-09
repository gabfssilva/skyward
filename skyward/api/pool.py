"""Public Pool protocol — the interface returned by Compute and Session."""

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
    Dispatch tasks via operators or methods.

    Examples
    --------
    >>> with sky.Compute(provider=sky.AWS(), accelerator="A100", nodes=4) as pool:
    ...     result = train(data) >> pool
    ...     results = train(data) @ pool
    ...     a, b = (task1() & task2()) >> pool
    ...     future = train(data) > pool
    """

    @property
    def concurrency(self) -> int:
        """Number of concurrent task slots per node."""
        ...

    @property
    def is_active(self) -> bool:
        """True if pool is ready for execution."""
        ...

    def run[T](self, pending: PendingFunction[T]) -> T:
        """Execute on one node (round-robin). Behind ``task() >> pool``."""
        ...

    def run_async[T](self, pending: PendingFunction[T]) -> Future[T]:
        """Submit asynchronously. Behind ``task() > pool``."""
        ...

    def broadcast[T](self, pending: PendingFunction[T]) -> list[T]:
        """Execute on ALL nodes. Behind ``task() @ pool``."""
        ...

    def run_parallel(
        self, group: PendingFunctionGroup,
    ) -> tuple[Any, ...] | Generator[Any, None, None]:
        """Execute a group concurrently. Behind ``(a() & b()) >> pool``."""
        ...

    def map[T, R](
        self, fn: Callable[[T], R], items: Sequence[T],
    ) -> list[R]:
        """Apply a function to each item, distributing across nodes."""
        ...

    def current_nodes(self) -> int:
        """Return the number of ready nodes."""
        ...

    def dict(self, name: str, *, consistency: Consistency | None = None) -> DictProxy:
        """Get or create a distributed dictionary."""
        ...

    def set(self, name: str, *, consistency: Consistency | None = None) -> SetProxy:
        """Get or create a distributed set."""
        ...

    def counter(self, name: str, *, consistency: Consistency | None = None) -> CounterProxy:
        """Get or create a distributed counter."""
        ...

    def queue(self, name: str) -> QueueProxy:
        """Get or create a distributed queue."""
        ...

    def barrier(self, name: str, n: int) -> BarrierProxy:
        """Get or create a distributed barrier."""
        ...

    def lock(self, name: str, timeout: float = 30) -> LockProxy:
        """Get or create a distributed lock."""
        ...
