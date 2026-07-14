"""What the user writes, and what it does not do.

``@function`` builds nothing but a description of a call. No pickling, no HTTP,
no compute: a ``Pending`` is inert until an operator hands it to a pool, which
is what lets the same call be dispatched to one node, to all of them, or not at
all.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Protocol


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


def function[**P, T](fn: Callable[P, T]) -> Callable[P, Pending[T]]:
    """Turn a function into one that describes a call instead of making it."""

    def pending(*args: P.args, **kwargs: P.kwargs) -> Pending[T]:
        return Pending(fn, args, kwargs)

    return pending
