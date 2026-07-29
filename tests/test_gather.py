"""``gather`` streaming and ordering, against a fake pool.

The fake borrows the real ``Compute.gather`` / ``Compute.gather_stream`` so the
ordering under test is the shipped one; only ``start`` is swapped for a thread
pool, which is all those methods touch. No control plane is provisioned.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor

from skyward.core.compute import Compute
from skyward.core.function import Group, Pending, gather


class FakePool:
    """A pool that runs each call on a thread pool and nothing more."""

    def __init__(self) -> None:
        self._exec = ThreadPoolExecutor(max_workers=8)

    def start[T](self, pending: Pending[T]) -> Future[T]:
        return self._exec.submit(pending.fn, *pending.args, **pending.kwargs)

    gather = Compute.gather
    gather_stream = Compute.gather_stream


def _call[T](fn) -> Pending[T]:  # noqa: ANN001
    return Pending(fn, (), {})


def test_gather_defaults() -> None:
    group = gather(_call(lambda: 1), _call(lambda: 2))
    assert isinstance(group, Group)
    assert group.stream is False
    assert group.ordered is True


def test_gather_returns_a_list_by_default() -> None:
    result = gather(_call(lambda: 1), _call(lambda: 2)) >> FakePool()
    assert result == [1, 2]


def test_and_keeps_list_semantics() -> None:
    result = (_call(lambda: "a") & _call(lambda: "b")) >> FakePool()
    assert result == ["a", "b"]


def test_stream_unordered_yields_as_they_finish() -> None:
    gate = threading.Event()

    def slow() -> str:
        gate.wait()
        return "slow"

    group = gather(_call(slow), _call(lambda: "fast"), stream=True, ordered=False)
    items = group >> FakePool()
    assert isinstance(items, Iterator)

    assert next(items) == "fast"
    gate.set()
    assert next(items) == "slow"


def test_stream_ordered_yields_in_submission_order() -> None:
    gate = threading.Event()

    def first() -> str:
        gate.wait()
        return "first"

    group = gather(_call(first), _call(lambda: "second"), stream=True, ordered=True)
    items = group >> FakePool()
    assert isinstance(items, Iterator)

    gate.set()
    assert list(items) == ["first", "second"]
