"""What the user writes, and what it does not do.

``@function`` builds nothing but a description of a call. No pickling, no HTTP,
no compute: a ``Pending`` is inert until an operator hands it to a pool, which
is what lets the same call be dispatched to one node, to all of them, or not at
all.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, replace
from typing import Protocol, overload


class Pool(Protocol):
    def run[T](self, pending: Pending[T]) -> T: ...

    def broadcast[T](self, pending: Pending[T]) -> list[T]: ...

    def start[T](self, pending: Pending[T]) -> Future[T]: ...

    def gather[T](self, group: Group[T]) -> list[T]: ...


@dataclass(frozen=True, slots=True)
class Pending[T]:
    fn: Callable[..., T]
    args: tuple[object, ...]
    kwargs: dict[str, object]
    timeout: float | None = None

    def with_timeout(self, timeout: float) -> Pending[T]:
        return replace(self, timeout=timeout)

    def __rshift__(self, pool: Pool) -> T:
        return pool.run(self)

    def __matmul__(self, pool: Pool) -> list[T]:
        return pool.broadcast(self)

    def __gt__(self, pool: Pool) -> Future[T]:
        return pool.start(self)

    def __and__(self, other: Pending[T]) -> Group[T]:
        return Group((self, other))


@dataclass(frozen=True, slots=True)
class Group[T]:
    """Calls that go together.

    Typed by what they return in common: ``a & b`` where both give an ``int`` is
    a ``Group[int]``. Mixing return types is allowed and lands on ``object`` —
    the group is honest about what it can promise rather than pretending to know
    which slot holds which type.
    """

    pendings: tuple[Pending[T], ...]

    def __and__(self, other: Pending[T]) -> Group[T]:
        return Group((*self.pendings, other))

    def __rshift__(self, pool: Pool) -> list[T]:
        return pool.gather(self)


def gather[T](*pendings: Pending[T]) -> Group[T]:
    """The same thing ``&`` builds, for when there are more than a few."""
    return Group(pendings)


@overload
def function[**P, T](fn: Callable[P, T]) -> Callable[P, Pending[T]]: ...


@overload
def function[**P, T](*, timeout: float) -> Callable[[Callable[P, T]], Callable[P, Pending[T]]]: ...


def function[**P, T](
    fn: Callable[P, T] | None = None,
    *,
    timeout: float | None = None,
) -> Callable[P, Pending[T]] | Callable[[Callable[P, T]], Callable[P, Pending[T]]]:
    """Turn a function into one that describes a call instead of making it.

    Bare (``@function``) or with a default timeout (``@function(timeout=600)``),
    which any single call can override with ``.with_timeout``.
    """

    def decorate(target: Callable[P, T]) -> Callable[P, Pending[T]]:
        def pending(*args: P.args, **kwargs: P.kwargs) -> Pending[T]:
            return Pending(target, args, kwargs, timeout)

        return pending

    return decorate(fn) if fn else decorate
