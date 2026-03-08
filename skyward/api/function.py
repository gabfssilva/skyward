from __future__ import annotations

import functools
import types
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, overload

from .context import _Sky


@dataclass(frozen=True, slots=True)
class PendingFunction[T]:
    """Lazy computation wrapper.

    Represents a function call that will be executed remotely
    when sent to a pool via the >> or @ operator.

    Example:
        @compute
        def train(data):
            return model.fit(data)

        pending = train(data)  # Returns PendingCompute, doesn't execute
        result = pending >> sky  # Executes remotely on pool
        results = pending @ sky  # Broadcasts to all nodes

        # Override timeout
        result = train(data).with_timeout(600) >> sky
    """

    fn: Callable[..., T]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    timeout: float | None = None

    def with_timeout(self, timeout: float) -> PendingFunction[T]:
        return PendingFunction(fn=self.fn, args=self.args, kwargs=self.kwargs, timeout=timeout)

    def __rshift__(self, target: Any) -> T:
        match target:
            case types.ModuleType() if hasattr(target, "sky"):
                return target.sky.__rrshift__(self)  # type: ignore
            case _Sky():
                return target.__rrshift__(self)  # type: ignore
            case _:
                return target.run(self)  # type: ignore[union-attr]

    def __gt__(self, target: Any) -> Future[T]:
        match target:
            case types.ModuleType() if hasattr(target, "sky"):
                return target.sky._run_async(self)  # type: ignore
            case _Sky():
                return target._run_async(self)  # type: ignore
            case _:
                return target.run_async(self)  # type: ignore[union-attr]

    def __matmul__(self, target: Any) -> list[T] | tuple[T, ...]:
        """Broadcast to all nodes using @ operator."""
        match target:
            case types.ModuleType() if hasattr(target, "sky"):
                return target.sky.__rmatmul__(self)
            case _Sky():
                return target.__rmatmul__(self)
            case _:
                return target.broadcast(self)  # type: ignore[union-attr]

    def __and__(self, other: PendingFunction[Any] | PendingFunctionGroup) -> PendingFunctionGroup:
        """Combine with another computation for parallel execution."""
        match other:
            case PendingFunctionGroup():
                return PendingFunctionGroup(items=(self, *other.items))
            case _:
                return PendingFunctionGroup(items=(self, other))


@dataclass(frozen=True, slots=True)
class PendingFunctionGroup:
    """Group of computations for parallel execution.

    Created by using the & operator:
        group = task1() & task2() & task3()
        a, b, c = group >> sky

    Or using gather():
        group = gather(task1(), task2(), task3())
        results = group >> sky
    """

    items: tuple[PendingFunction[Any], ...]
    stream: bool = False
    ordered: bool = True
    timeout: float | None = None

    def with_timeout(self, timeout: float) -> PendingFunctionGroup:
        return PendingFunctionGroup(
            items=self.items, stream=self.stream,
            ordered=self.ordered, timeout=timeout,
        )

    def __and__(self, other: PendingFunction[Any] | PendingFunctionGroup) -> PendingFunctionGroup:
        """Add another computation to the group."""
        match other:
            case PendingFunctionGroup():
                return PendingFunctionGroup(
                    items=(*self.items, *other.items),
                    stream=self.stream, ordered=self.ordered,
                )
            case _:
                return PendingFunctionGroup(
                    items=(*self.items, other),
                    stream=self.stream, ordered=self.ordered,
                )

    def __rshift__(self, target: Any) -> tuple[Any, ...] | Any:
        """Execute all computations in parallel using >> operator."""
        match target:
            case types.ModuleType() if hasattr(target, "sky"):
                return target.sky.__rrshift__(self)  # type: ignore
            case _Sky():
                return target.__rrshift__(self)  # type: ignore
            case _:
                return target.run_parallel(self)  # type: ignore[union-attr]

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[PendingFunction[Any]]:
        return iter(self.items)


def gather(
    *pendings: PendingFunction[Any],
    stream: bool = False,
    ordered: bool = True,
) -> PendingFunctionGroup:
    """Group computations for parallel execution.

    Example:
        results = gather(task1(), task2(), task3()) >> sky
        # results is a tuple of (result1, result2, result3)

        for result in gather(task1(), task2(), task3(), stream=True) >> sky:
            print(result)  # yields results as they complete

        for result in gather(task1(), task2(), task3(), stream=True, ordered=False) >> sky:
            print(result)  # yields results as they complete, unordered
    """
    return PendingFunctionGroup(items=pendings, stream=stream, ordered=ordered)


@overload
def function[**P, T](fn: Callable[P, T]) -> Callable[P, PendingFunction[T]]: ...

@overload
def function[**P, T](
    *, timeout: float,
) -> Callable[[Callable[P, T]], Callable[P, PendingFunction[T]]]: ...

def function[**P, T](
    fn: Callable[P, T] | None = None,
    *,
    timeout: float | None = None,
) -> Callable[P, PendingFunction[T]] | Callable[[Callable[P, T]], Callable[P, PendingFunction[T]]]:
    def decorator(f: Callable[P, T]) -> Callable[P, PendingFunction[T]]:
        @functools.wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> PendingFunction[T]:
            return PendingFunction(fn=f, args=args, kwargs=kwargs, timeout=timeout)
        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator
