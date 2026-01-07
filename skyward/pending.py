"""Lazy computation layer for skyward.

This module provides the lazy evaluation primitives:
- PendingCompute[R]: A deferred computation that hasn't been executed yet
- PendingBatch: Groups multiple PendingCompute for parallel execution
- gather(): Groups computations with type-safe overloads
- compute: Standalone decorator that creates lazy functions

Example:
    from skyward import compute, gather, Pool, AWS

    @compute
    def train(data):
        return model.fit(data)

    @compute
    def evaluate(model):
        return model.evaluate()

    pool = Pool(provider=AWS(), pip=["torch"])

    # Single execution
    result = train(data) | pool

    # Parallel execution
    r1, r2 = gather(train(data1), train(data2)) | pool
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from types import ModuleType
from typing import TYPE_CHECKING, Any, TypeVar, overload

if TYPE_CHECKING:
    from skyward.pool import ComputePool

# Type alias for pool target (explicit pool or implicit via module)
type PoolTarget = ComputePool | ModuleType


def _run_batch(batch: Any, target: PoolTarget) -> tuple[Any, ...]:
    """Execute a typed batch on the given pool."""
    computations = tuple(getattr(batch, f.name) for f in fields(batch))
    return _resolve_pool(target).run_batch(PendingBatch(computations))


def _resolve_pool(target: PoolTarget) -> ComputePool:
    """Resolve pool from target (explicit pool or implicit via module context)."""
    match target:
        case ModuleType():
            from skyward._context import get_current_pool

            return get_current_pool()
        case _:
            return target


# Type variables for generic return types
R = TypeVar("R")
R1 = TypeVar("R1")
R2 = TypeVar("R2")
R3 = TypeVar("R3")
R4 = TypeVar("R4")
R5 = TypeVar("R5")
R6 = TypeVar("R6")
R7 = TypeVar("R7")
R8 = TypeVar("R8")


@dataclass(frozen=True, slots=True)
class PendingCompute[R]:
    """A deferred computation that hasn't been executed yet.

    Created by calling a @compute decorated function. Execution is
    deferred until the computation is sent to a Pool using the pipe
    operator (|).

    Type Parameters:
        R: The return type of the computation

    Example:
        @compute
        def process(x: int) -> float:
            return x * 1.5

        pending = process(10)  # Returns PendingCompute[float]
        result = pending | pool  # Executes and returns float
    """

    fn: Callable[..., R]
    args: tuple[Any, ...]
    kwargs: frozenset[tuple[str, Any]]
    name: str = ""

    def __and__[R2](self, other: PendingCompute[R2]) -> PendingBatch2[R, R2]:
        """Chain computations for parallel execution.

        Example:
            first, second = add(1, 2) & add(3, 4) | pool
            a, b, c = add(1, 2) & add(3, 4) & add(5, 6) | pool
        """
        return PendingBatch2(self, other)

    def __rshift__(self, target: PoolTarget) -> R:
        """Execute this computation on the given pool.

        Args:
            target: Pool to execute on, or skyward module for implicit pool.

        Returns:
            The result of the computation.
        """
        pool = _resolve_pool(target)
        return pool.run(self)

    def __matmul__(self, target: PoolTarget) -> tuple[R, ...]:
        """Broadcast: execute on ALL nodes in the pool.

        Args:
            target: Pool to execute on, or skyward module for implicit pool.

        Returns:
            Tuple of results, one per node.

        Example:
            # Execute on all 4 nodes
            results = load_model(path) @ pool  # tuple of 4 results
        """
        return _resolve_pool(target).broadcast(self)

    @property
    def kwargs_dict(self) -> dict[str, Any]:
        """Return kwargs as a regular dict."""
        return dict(self.kwargs)


@dataclass(frozen=True, slots=True)
class PendingBatch:
    """A batch of deferred computations for parallel execution.

    Created by the gather() function. When sent to a Pool using the
    pipe operator, all computations execute in parallel.

    Example:
        batch = gather(fn1(x), fn2(y), fn3(z))
        r1, r2, r3 = batch | pool  # Parallel execution

        # Or using & operator (returns typed PendingBatch2/3/etc):
        r1, r2, r3 = fn1(x) & fn2(y) & fn3(z) | pool
    """

    computations: tuple[PendingCompute[Any], ...]

    def __rshift__(self, target: PoolTarget) -> tuple[Any, ...]:
        """Execute all computations in parallel on the given pool.

        Args:
            target: Pool to execute on, or skyward module for implicit pool.

        Returns:
            Tuple of results in the same order as the computations.
        """
        return _resolve_pool(target).run_batch(self)


# =============================================================================
# Typed PendingBatch classes for & operator
# =============================================================================


@dataclass(frozen=True, slots=True)
class PendingBatch2[R1, R2]:
    """Typed batch of 2 computations."""

    c1: PendingCompute[R1]
    c2: PendingCompute[R2]

    def __and__[R3](self, other: PendingCompute[R3]) -> PendingBatch3[R1, R2, R3]:
        return PendingBatch3(self.c1, self.c2, other)

    def __rshift__(self, target: PoolTarget) -> tuple[R1, R2]:
        return _run_batch(self, target)


@dataclass(frozen=True, slots=True)
class PendingBatch3[R1, R2, R3]:
    """Typed batch of 3 computations."""

    c1: PendingCompute[R1]
    c2: PendingCompute[R2]
    c3: PendingCompute[R3]

    def __and__[R4](self, other: PendingCompute[R4]) -> PendingBatch4[R1, R2, R3, R4]:
        return PendingBatch4(self.c1, self.c2, self.c3, other)

    def __rshift__(self, target: PoolTarget) -> tuple[R1, R2, R3]:
        return _run_batch(self, target)


@dataclass(frozen=True, slots=True)
class PendingBatch4[R1, R2, R3, R4]:
    """Typed batch of 4 computations."""

    c1: PendingCompute[R1]
    c2: PendingCompute[R2]
    c3: PendingCompute[R3]
    c4: PendingCompute[R4]

    def __and__[R5](self, other: PendingCompute[R5]) -> PendingBatch5[R1, R2, R3, R4, R5]:
        return PendingBatch5(self.c1, self.c2, self.c3, self.c4, other)

    def __rshift__(self, target: PoolTarget) -> tuple[R1, R2, R3, R4]:
        return _run_batch(self, target)


@dataclass(frozen=True, slots=True)
class PendingBatch5[R1, R2, R3, R4, R5]:
    """Typed batch of 5 computations."""

    c1: PendingCompute[R1]
    c2: PendingCompute[R2]
    c3: PendingCompute[R3]
    c4: PendingCompute[R4]
    c5: PendingCompute[R5]

    def __and__[R6](self, other: PendingCompute[R6]) -> PendingBatch6[R1, R2, R3, R4, R5, R6]:
        return PendingBatch6(self.c1, self.c2, self.c3, self.c4, self.c5, other)

    def __rshift__(self, target: PoolTarget) -> tuple[R1, R2, R3, R4, R5]:
        return _run_batch(self, target)


@dataclass(frozen=True, slots=True)
class PendingBatch6[R1, R2, R3, R4, R5, R6]:
    """Typed batch of 6 computations."""

    c1: PendingCompute[R1]
    c2: PendingCompute[R2]
    c3: PendingCompute[R3]
    c4: PendingCompute[R4]
    c5: PendingCompute[R5]
    c6: PendingCompute[R6]

    def __and__[R7](self, other: PendingCompute[R7]) -> PendingBatch7[R1, R2, R3, R4, R5, R6, R7]:
        return PendingBatch7(self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, other)

    def __rshift__(self, target: PoolTarget) -> tuple[R1, R2, R3, R4, R5, R6]:
        return _run_batch(self, target)


@dataclass(frozen=True, slots=True)
class PendingBatch7[R1, R2, R3, R4, R5, R6, R7]:
    """Typed batch of 7 computations."""

    c1: PendingCompute[R1]
    c2: PendingCompute[R2]
    c3: PendingCompute[R3]
    c4: PendingCompute[R4]
    c5: PendingCompute[R5]
    c6: PendingCompute[R6]
    c7: PendingCompute[R7]

    def __and__[R8](
        self, other: PendingCompute[R8]
    ) -> PendingBatch8[R1, R2, R3, R4, R5, R6, R7, R8]:
        return PendingBatch8(self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, self.c7, other)

    def __rshift__(self, target: PoolTarget) -> tuple[R1, R2, R3, R4, R5, R6, R7]:
        return _run_batch(self, target)


@dataclass(frozen=True, slots=True)
class PendingBatch8[R1, R2, R3, R4, R5, R6, R7, R8]:
    """Typed batch of 8 computations."""

    c1: PendingCompute[R1]
    c2: PendingCompute[R2]
    c3: PendingCompute[R3]
    c4: PendingCompute[R4]
    c5: PendingCompute[R5]
    c6: PendingCompute[R6]
    c7: PendingCompute[R7]
    c8: PendingCompute[R8]

    def __rshift__(self, target: PoolTarget) -> tuple[R1, R2, R3, R4, R5, R6, R7, R8]:
        return _run_batch(self, target)


# =============================================================================
# gather() with type-safe overloads
# =============================================================================


@overload
def gather(c1: PendingCompute[R1], /) -> PendingBatch: ...


@overload
def gather(
    c1: PendingCompute[R1],
    c2: PendingCompute[R2],
    /,
) -> PendingBatch: ...


@overload
def gather(
    c1: PendingCompute[R1],
    c2: PendingCompute[R2],
    c3: PendingCompute[R3],
    /,
) -> PendingBatch: ...


@overload
def gather(
    c1: PendingCompute[R1],
    c2: PendingCompute[R2],
    c3: PendingCompute[R3],
    c4: PendingCompute[R4],
    /,
) -> PendingBatch: ...


@overload
def gather(
    c1: PendingCompute[R1],
    c2: PendingCompute[R2],
    c3: PendingCompute[R3],
    c4: PendingCompute[R4],
    c5: PendingCompute[R5],
    /,
) -> PendingBatch: ...


@overload
def gather(
    c1: PendingCompute[R1],
    c2: PendingCompute[R2],
    c3: PendingCompute[R3],
    c4: PendingCompute[R4],
    c5: PendingCompute[R5],
    c6: PendingCompute[R6],
    /,
) -> PendingBatch: ...


@overload
def gather(
    c1: PendingCompute[R1],
    c2: PendingCompute[R2],
    c3: PendingCompute[R3],
    c4: PendingCompute[R4],
    c5: PendingCompute[R5],
    c6: PendingCompute[R6],
    c7: PendingCompute[R7],
    /,
) -> PendingBatch: ...


@overload
def gather(
    c1: PendingCompute[R1],
    c2: PendingCompute[R2],
    c3: PendingCompute[R3],
    c4: PendingCompute[R4],
    c5: PendingCompute[R5],
    c6: PendingCompute[R6],
    c7: PendingCompute[R7],
    c8: PendingCompute[R8],
    /,
) -> PendingBatch: ...


@overload
def gather(*computations: PendingCompute[Any]) -> PendingBatch: ...


def gather(*computations: PendingCompute[Any]) -> PendingBatch:
    """Group multiple computations for parallel execution.

    When the resulting PendingBatch is sent to a Pool, all computations
    execute in parallel, and the results are returned as a tuple.

    Args:
        *computations: PendingCompute objects to group.

    Returns:
        PendingBatch that can be sent to a Pool.

    Example:
        # Type inference works for up to 8 computations
        r1, r2 = gather(fn1(x), fn2(y)) | pool

        # Works with any number of computations
        results = gather(*[fn(i) for i in range(100)]) | pool
    """
    return PendingBatch(computations=computations)


# =============================================================================
# @compute decorator
# =============================================================================


@dataclass(frozen=True, slots=True)
class ComputeFunction[**P, R]:
    """A function that has been decorated with @compute.

    When called, returns a PendingCompute instead of executing immediately.
    The original function is preserved and can be accessed via .local.
    """

    fn: Callable[P, R]
    name: str = ""

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> PendingCompute[R]:
        """Create a PendingCompute from this function call.

        Args:
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            PendingCompute that can be sent to a Pool for execution.
        """
        return PendingCompute(
            fn=self.fn,
            args=args,
            kwargs=frozenset(kwargs.items()),
            name=self.name or self.fn.__name__,
        )

    @property
    def local(self) -> Callable[P, R]:
        """Return the original function for local execution."""
        return self.fn

    def __repr__(self) -> str:
        name = self.name or self.fn.__name__
        return f"ComputeFunction({name})"


def compute[**P, R](fn: Callable[P, R]) -> Callable[P, PendingCompute[R]]:
    """Decorator that makes a function lazy.

    When a @compute decorated function is called, it returns a
    PendingCompute instead of executing immediately. The computation
    is executed when sent to a Pool using the pipe operator (|).

    Args:
        fn: Function to make lazy.

    Returns:
        ComputeFunction that creates PendingCompute on call.

    Example:
        @compute
        def train(data: list[int]) -> float:
            return sum(data) / len(data)

        # Returns PendingCompute[float], not float
        pending = train([1, 2, 3, 4])

        # Execute on a pool
        result = pending | pool  # Returns 2.5
    """
    return ComputeFunction(fn=fn, name=fn.__name__)
