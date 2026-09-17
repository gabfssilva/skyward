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
from types import ModuleType
from typing import Protocol, overload

from msgspec import UNSET, UnsetType

from skyward.core import context
from skyward.core.context import _Sky
from skyward.shared.retry import Retry


class Pool(Protocol):
    def run[T](self, pending: Pending[T]) -> T: ...

    def broadcast[T](self, pending: Pending[T]) -> list[T]: ...

    def start[T](self, pending: Pending[T]) -> Future[T]: ...

    def gather[T](self, group: Group[T]) -> list[T]: ...

    def gather_stream[T](self, group: Group[T]) -> Iterator[T]: ...

    def stream[T](self, pending: Streaming[T]) -> Iterator[T]: ...


type Target = Pool | _Sky | ModuleType
"""Where a call goes: a named pool, the ``sky`` stand-in, or the ``skyward``
module itself (what ``import skyward as sky`` binds ``sky`` to)."""


def _pool(target: Target) -> Pool:
    """The pool to dispatch to, resolving the implicit target from context."""
    match target:
        case _Sky() | ModuleType():
            return context.current()
        case _:
            return target


@dataclass(frozen=True, slots=True)
class Pending[T]:
    """One call, described and not yet made.

    ``timeout`` is how long each attempt may run once started, and ``queue_timeout``
    how long it may wait for a machine before it starts: ``None`` takes the pool's
    ``task_run_timeout`` and ``task_queue_timeout``, and ``0`` is no limit whatever
    the pool says. ``retry`` is the call's retry decision: unset takes the pool's,
    ``None`` turns retrying off for this call, and a function decides — see
    :func:`function`.
    """

    fn: Callable[..., T]
    args: tuple[object, ...]
    kwargs: dict[str, object]
    timeout: float | None = None
    retry: Retry | None | UnsetType = UNSET
    queue_timeout: float | None = None

    def with_timeout(self, timeout: float) -> Pending[T]:
        return replace(self, timeout=timeout)

    def with_queue_timeout(self, queue_timeout: float) -> Pending[T]:
        return replace(self, queue_timeout=queue_timeout)

    def with_retry(self, retry: Retry | None) -> Pending[T]:
        return replace(self, retry=retry)

    def __rshift__(self, target: Target) -> T:
        return _pool(target).run(self)

    def __matmul__(self, target: Target) -> list[T]:
        return _pool(target).broadcast(self)

    def __gt__(self, target: Target) -> Future[T]:
        return _pool(target).start(self)

    def __and__(self, other: Pending[T]) -> Group[T]:
        return Group((self, other))


@dataclass(frozen=True, slots=True)
class Group[T]:
    """Calls that go together.

    Typed by what they return in common: ``a & b`` where both give an ``int`` is
    a ``Group[int]``. Mixing return types is allowed and lands on ``object`` —
    the group is honest about what it can promise rather than pretending to know
    which slot holds which type.

    ``stream`` changes what ``>>`` gives back: a list once every call is in, or an
    iterator that hands over each result the moment it is ready. ``ordered`` picks
    between the two ways to be early — submission order, blocking only on the next
    one due, or completion order, whichever finishes first.
    """

    pendings: tuple[Pending[T], ...]
    stream: bool = False
    ordered: bool = True

    def __and__(self, other: Pending[T]) -> Group[T]:
        return Group((*self.pendings, other), self.stream, self.ordered)

    def __rshift__(self, target: Target) -> list[T] | Iterator[T]:
        pool = _pool(target)
        return pool.gather_stream(self) if self.stream else pool.gather(self)


@dataclass(frozen=True, slots=True)
class Streaming[T]:
    """A call whose answer arrives in pieces.

    ``timeout`` is how long the stream may run once started — ``None`` takes the
    pool's ``task_run_timeout``, and ``0`` is no limit.

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

    def __rshift__(self, target: Target) -> Iterator[T]:
        return _pool(target).stream(self)


def gather[T](*pendings: Pending[T], stream: bool = False, ordered: bool = True) -> Group[T]:
    """The same thing ``&`` builds, for when there are more than a few.

    ``stream`` turns ``>>`` from a list into an iterator that yields each result as
    it lands; ``ordered`` keeps that iterator in submission order, waiting on the
    next one due, rather than in completion order. Both are inert until dispatched.
    """
    return Group(pendings, stream, ordered)


@overload
def function[**P, T](fn: Callable[P, T]) -> Callable[P, Pending[T]]: ...


@overload
def function[**P, T](
    *, timeout: float | None = None, queue_timeout: float | None = None, retry: Retry | None | UnsetType = UNSET
) -> Callable[[Callable[P, T]], Callable[P, Pending[T]]]: ...


def function[**P, T](
    fn: Callable[P, T] | None = None,
    *,
    timeout: float | None = None,
    queue_timeout: float | None = None,
    retry: Retry | None | UnsetType = UNSET,
) -> Callable[P, Pending[T]] | Callable[[Callable[P, T]], Callable[P, Pending[T]]]:
    """Turn a function into one that describes a call instead of making it.

    Bare (``@function``) or with defaults (``@function(timeout=600)``), which any
    single call can override with ``.with_timeout``, ``.with_queue_timeout`` and
    ``.with_retry``.

    ``timeout`` is how long each attempt may run, counted from when it starts on a
    machine — not from the submission, so a call behind a long queue keeps all of
    it, and a retry starts the count again. One that runs past it is stopped on the
    machine and ends ``timed_out``. ``queue_timeout`` is how long each attempt may
    wait for a machine before it starts. ``None`` takes the pool's
    ``task_run_timeout`` and ``task_queue_timeout``; ``0`` is no limit.

    ``retry`` is a ``(reason, attempt) -> bool`` asked when an attempt does not
    answer. ``reason`` is the exception the function raised, or a :class:`sky.Lost`
    when the attempt was lost — the process died under it, the worker restarted, the
    machine went away — and ``attempt`` is the one that just failed, from one. An
    exception is asked about on the node, where it was raised; a loss on the daemon.
    Left unset, the call takes the pool's decision, whose default tries once more
    after a loss and never after an exception: a function that raised is retried
    only if the decision says so. ``None`` turns retrying off for this function.
    """

    def decorate(target: Callable[P, T]) -> Callable[P, Pending[T]]:
        if inspect.isgeneratorfunction(target):
            raise TypeError(f"{target.__name__} is a generator: decorate it with @stream, which gives back its items")

        def pending(*args: P.args, **kwargs: P.kwargs) -> Pending[T]:
            return Pending(target, args, kwargs, timeout, retry, queue_timeout)

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
