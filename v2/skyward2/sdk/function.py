"""What the user writes, and what it does not do.

``@function`` builds nothing but a description of a call. No pickling, no HTTP,
no compute: a ``Pending`` is inert until an operator hands it to a pool, which
is what lets the same call be dispatched to one node, to all of them, or not at
all.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from dataclasses import dataclass, replace
from typing import Protocol, overload


class Pool(Protocol):
    def run[T](self, pending: Pending[T]) -> T: ...

    def broadcast[T](self, pending: Pending[T]) -> list[T]: ...

    def start[T](self, pending: Pending[T]) -> Future[T]: ...

    def gather[T](self, group: Group[T]) -> list[T]: ...

    def stream[T](self, pending: Streaming[T]) -> Iterator[T]: ...


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


@dataclass(frozen=True, slots=True)
class Streaming[T]:
    """A call whose answer arrives in pieces.

    What a generator function becomes. It is a separate type from ``Pending`` and
    not a flag on it, because it is a separate promise: ``>>`` gives back an
    iterator here, and the difference is worth knowing before the code runs rather
    than after — a generator dispatched as an ordinary call would pickle the
    generator object and fail on the machine.
    """

    fn: Callable[..., Iterator[T]]
    args: tuple[object, ...]
    kwargs: dict[str, object]
    timeout: float | None = None

    def with_timeout(self, timeout: float) -> Streaming[T]:
        return replace(self, timeout=timeout)

    def __rshift__(self, pool: Pool) -> Iterator[T]:
        return pool.stream(self)


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
        if inspect.isgeneratorfunction(target):
            raise TypeError(f"{target.__name__} is a generator: decorate it with @stream, which gives back its items")

        def pending(*args: P.args, **kwargs: P.kwargs) -> Pending[T]:
            return Pending(target, args, kwargs, timeout)

        return pending

    return decorate(fn) if fn else decorate


@overload
def stream[**P, T](fn: Callable[P, Iterator[T]]) -> Callable[P, Streaming[T]]: ...


@overload
def stream[**P, T](*, timeout: float) -> Callable[[Callable[P, Iterator[T]]], Callable[P, Streaming[T]]]: ...


def stream[**P, T](
    fn: Callable[P, Iterator[T]] | None = None,
    *,
    timeout: float | None = None,
) -> Callable[P, Streaming[T]] | Callable[[Callable[P, Iterator[T]]], Callable[P, Streaming[T]]]:
    """Turn a generator into one that describes a stream instead of making it.

        @sky.stream
        def tokens(prompt: str) -> Iterator[str]:
            yield from model.generate(prompt)

        for token in tokens("hi") >> pool:
            print(token)

    A separate decorator rather than a flag on ``@function``, because it is a
    separate promise: ``>>`` hands back an iterator, and the items arrive as the
    machine produces them. Worth knowing where the function is defined rather than
    where it is called.
    """

    def decorate(target: Callable[P, Iterator[T]]) -> Callable[P, Streaming[T]]:
        def streaming(*args: P.args, **kwargs: P.kwargs) -> Streaming[T]:
            return Streaming(target, args, kwargs, timeout)

        return streaming

    return decorate(fn) if fn else decorate
