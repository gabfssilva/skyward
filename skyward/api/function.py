"""Lazy computation primitives and the ``@sky.function`` decorator.

Provides ``PendingFunction[T]`` — a frozen snapshot of a function call that
does nothing until dispatched to a pool via an operator.  The ``&`` operator
composes pending functions into ``PendingFunctionGroup`` for parallel
execution, and ``gather()`` offers the same capability as a function call.
"""

from __future__ import annotations

import functools
import types
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, overload

from skyward.api.context import _Sky


@dataclass(frozen=True, slots=True)
class PendingFunction[T]:
    """Lazy computation — a frozen snapshot of function + args.

    Created by ``@sky.function`` decorated calls. Nothing executes
    until dispatched to a pool via an operator (``>>``, ``@``, ``>``, ``&``).

    Examples
    --------
    >>> @sky.function
    ... def train(data):
    ...     return model.fit(data)

    >>> pending = train(data)  # Returns PendingFunction, doesn't execute
    >>> result = pending >> sky  # Execute on one node
    >>> results = pending @ sky  # Broadcast to all nodes
    >>> result = train(data).with_timeout(600) >> sky  # Override timeout
    """

    fn: Callable[..., T]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    timeout: float | None = None

    def with_timeout(self, timeout: float) -> PendingFunction[T]:
        """Return a copy with the given execution timeout in seconds.

        Parameters
        ----------
        timeout
            Maximum execution time in seconds.

        Returns
        -------
        PendingFunction[T]
            New instance with the timeout set.
        """
        return PendingFunction(fn=self.fn, args=self.args, kwargs=self.kwargs, timeout=timeout)

    def __rshift__(self, target: Any) -> T:
        """Execute on one node via ``task() >> pool``.

        Dispatch this pending function to *target* for synchronous
        execution on a single node (round-robin selection).

        Parameters
        ----------
        target
            A ``ComputePool``, the ``sky`` singleton, or the ``skyward`` module.

        Returns
        -------
        T
            The remote function's return value.
        """
        match target:
            case types.ModuleType() if hasattr(target, "sky"):
                return target.sky.__rrshift__(self)  # type: ignore
            case _Sky():
                return target.__rrshift__(self)  # type: ignore
            case _:
                return target.run(self)  # type: ignore[union-attr]

    def __gt__(self, target: Any) -> Future[T]:
        """Execute asynchronously via ``task() > pool``.

        Dispatch this pending function for non-blocking execution,
        returning a ``Future`` that resolves when the task completes.

        Parameters
        ----------
        target
            A ``ComputePool``, the ``sky`` singleton, or the ``skyward`` module.

        Returns
        -------
        Future[T]
            A future that resolves to the remote function's return value.
        """
        match target:
            case types.ModuleType() if hasattr(target, "sky"):
                return target.sky._run_async(self)  # type: ignore
            case _Sky():
                return target._run_async(self)  # type: ignore
            case _:
                return target.run_async(self)  # type: ignore[union-attr]

    def __matmul__(self, target: Any) -> list[T] | tuple[T, ...]:
        """Broadcast to all nodes via ``task() @ pool``.

        Dispatch this pending function to every node in the pool,
        returning one result per node.

        Parameters
        ----------
        target
            A ``ComputePool``, the ``sky`` singleton, or the ``skyward`` module.

        Returns
        -------
        list[T] | tuple[T, ...]
            One result per node in the pool.
        """
        match target:
            case types.ModuleType() if hasattr(target, "sky"):
                return target.sky.__rmatmul__(self)
            case _Sky():
                return target.__rmatmul__(self)
            case _:
                return target.broadcast(self)  # type: ignore[union-attr]

    def __and__(self, other: PendingFunction[Any] | PendingFunctionGroup) -> PendingFunctionGroup:
        """Combine with another pending function via ``task1() & task2()``.

        Create a ``PendingFunctionGroup`` for parallel execution when
        dispatched with ``>> pool``.

        Parameters
        ----------
        other
            Another pending function or group to combine with.

        Returns
        -------
        PendingFunctionGroup
            A group containing both operands.
        """
        match other:
            case PendingFunctionGroup():
                return PendingFunctionGroup(items=(self, *other.items))
            case _:
                return PendingFunctionGroup(items=(self, other))


@dataclass(frozen=True, slots=True)
class PendingFunctionGroup:
    """Group of pending functions for parallel execution.

    Created by chaining ``&`` operators or calling ``sky.gather()``.
    Dispatch the group with ``>> pool`` to run all tasks concurrently.

    Examples
    --------
    >>> group = task1() & task2() & task3()
    >>> a, b, c = group >> sky

    >>> group = sky.gather(task1(), task2(), task3())
    >>> results = group >> sky
    """

    items: tuple[PendingFunction[Any], ...]
    stream: bool = False
    ordered: bool = True
    timeout: float | None = None

    def with_timeout(self, timeout: float) -> PendingFunctionGroup:
        """Return a copy with the given execution timeout in seconds.

        Parameters
        ----------
        timeout
            Maximum execution time in seconds, applied to the group.

        Returns
        -------
        PendingFunctionGroup
            New instance with the timeout set.
        """
        return PendingFunctionGroup(
            items=self.items, stream=self.stream,
            ordered=self.ordered, timeout=timeout,
        )

    def __and__(self, other: PendingFunction[Any] | PendingFunctionGroup) -> PendingFunctionGroup:
        """Append another pending function or group via ``&``.

        Parameters
        ----------
        other
            Another pending function or group to add.

        Returns
        -------
        PendingFunctionGroup
            A new group containing all combined items.
        """
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
        """Execute all tasks in parallel via ``group >> pool``.

        Dispatch every pending function in the group concurrently
        and return their results as a tuple.

        Parameters
        ----------
        target
            A ``ComputePool``, the ``sky`` singleton, or the ``skyward`` module.

        Returns
        -------
        tuple[Any, ...] | Any
            Results from all tasks, one element per pending function.
        """
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
    """Group pending functions for parallel execution.

    Combine multiple ``PendingFunction`` instances into a
    ``PendingFunctionGroup`` that runs all tasks concurrently
    when dispatched with ``>> pool``.

    Parameters
    ----------
    *pendings
        Pending functions to execute in parallel.
    stream
        If ``True``, yield results as they complete instead of
        returning a tuple. Default ``False``.
    ordered
        If ``True``, preserve submission order when streaming.
        Ignored when *stream* is ``False``. Default ``True``.

    Returns
    -------
    PendingFunctionGroup
        A group ready for dispatch.

    Examples
    --------
    >>> results = gather(task1(), task2(), task3()) >> sky

    >>> for result in gather(task1(), task2(), stream=True) >> sky:
    ...     print(result)
    """
    return PendingFunctionGroup(items=pendings, stream=stream, ordered=ordered)


@overload
def function[**P, T](fn: Callable[P, T]) -> Callable[P, PendingFunction[T]]:
    """Mark a function for remote execution on a compute pool.

    Wrapping a function with ``@sky.function`` makes it return a
    ``PendingFunction[T]`` when called, capturing args without executing.
    Dispatch via operators: ``>> pool``, ``@ pool``, ``> pool``.

    Can be used bare (``@sky.function``) or with a default timeout
    (``@sky.function(timeout=600)``).

    Parameters
    ----------
    fn
        The function to wrap. Provided implicitly when used as
        ``@sky.function`` without parentheses.
    timeout
        Default execution timeout in seconds. Can be overridden
        per-call via ``PendingFunction.with_timeout``.

    Returns
    -------
    Callable[P, PendingFunction[T]]
        A wrapper that captures calls as pending functions.

    Examples
    --------
    >>> @sky.function
    ... def train(data):
    ...     return model.fit(data)

    >>> @sky.function(timeout=600)
    ... def long_train(data):
    ...     return model.fit(data)
    """
    ...

@overload
def function[**P, T](
    *, timeout: float,
) -> Callable[[Callable[P, T]], Callable[P, PendingFunction[T]]]:
    """Mark a function for remote execution on a compute pool.

    Wrapping a function with ``@sky.function`` makes it return a
    ``PendingFunction[T]`` when called, capturing args without executing.
    Dispatch via operators: ``>> pool``, ``@ pool``, ``> pool``.

    Can be used bare (``@sky.function``) or with a default timeout
    (``@sky.function(timeout=600)``).

    Parameters
    ----------
    fn
        The function to wrap. Provided implicitly when used as
        ``@sky.function`` without parentheses.
    timeout
        Default execution timeout in seconds. Can be overridden
        per-call via ``PendingFunction.with_timeout``.

    Returns
    -------
    Callable[P, PendingFunction[T]]
        A wrapper that captures calls as pending functions.

    Examples
    --------
    >>> @sky.function
    ... def train(data):
    ...     return model.fit(data)

    >>> @sky.function(timeout=600)
    ... def long_train(data):
    ...     return model.fit(data)
    """
    ...

def function[**P, T](
    fn: Callable[P, T] | None = None,
    *,
    timeout: float | None = None,
) -> Callable[P, PendingFunction[T]] | Callable[[Callable[P, T]], Callable[P, PendingFunction[T]]]:
    """Mark a function for remote execution on a compute pool.

    Wrapping a function with ``@sky.function`` makes it return a
    ``PendingFunction[T]`` when called, capturing args without executing.
    Dispatch via operators: ``>> pool``, ``@ pool``, ``> pool``.

    Can be used bare (``@sky.function``) or with a default timeout
    (``@sky.function(timeout=600)``).

    Parameters
    ----------
    fn
        The function to wrap. Provided implicitly when used as
        ``@sky.function`` without parentheses.
    timeout
        Default execution timeout in seconds. Can be overridden
        per-call via ``PendingFunction.with_timeout``.

    Returns
    -------
    Callable[P, PendingFunction[T]]
        A wrapper that captures calls as pending functions.

    Examples
    --------
    >>> @sky.function
    ... def train(data):
    ...     return model.fit(data)

    >>> @sky.function(timeout=600)
    ... def long_train(data):
    ...     return model.fit(data)
    """
    def decorator(f: Callable[P, T]) -> Callable[P, PendingFunction[T]]:
        @functools.wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> PendingFunction[T]:
            return PendingFunction(fn=f, args=args, kwargs=kwargs, timeout=timeout)
        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator
