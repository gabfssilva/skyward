"""The bridge, exercised with a real subprocess but no cluster.

A collection call made inside a process-pool worker has to travel a pipe back to the
parent and be answered there. This stands a real executor behind the bridge and puts a
fake backend where casty would be, so the round trip — the Call over the pipe, the
bridge waking, the answer coming home — is what gets tested, not casty.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from skyward import Executor, distributed
from skyward.runtime import ipc


class Recorder:
    """Stands in for the worker's cluster: records every call, keeps one counter."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, tuple[object, ...]]] = []
        self.total = 0

    def __call__(self, kind: str, name: str, method: str, args: tuple[object, ...], params: object) -> object:
        self.calls.append((kind, name, method, args))
        match (kind, method):
            case ("counter", "add"):
                delta = args[0]
                assert isinstance(delta, int)
                self.total += delta
                return None
            case ("counter", "get"):
                return self.total
            case ("lock", "acquire"):
                return "token-1"
            case _:
                return None


def _bump_and_read() -> int:
    """Runs in a subprocess: reaches a counter it cannot see, over the bridge."""
    counter = distributed.counter("processed")
    counter.add(5)
    return counter.get()


def _take_the_lock() -> bool:
    """Runs in a subprocess: acquires and releases a lock held by the parent."""
    with distributed.lock("checkpoint"):
        return True


@pytest.fixture
def recorded() -> Iterator[Recorder]:
    recorder = Recorder()
    distributed.install(recorder)
    try:
        yield recorder
    finally:
        distributed.install(distributed._local)


@pytest.mark.parametrize(
    "executor",
    [Executor("process", True), Executor("process", False), Executor("loky", True)],
    ids=["process-reused", "process-fresh", "loky"],
)
def test_a_counter_touched_in_a_subprocess_reaches_the_parent(recorded: Recorder, executor: Executor) -> None:
    with ipc.executor(executor.type, reuse=executor.reuse, workers=1) as pool:
        result = pool.submit(_bump_and_read).result(timeout=60)

    assert result == 5, "the subprocess read the value the parent held"
    assert ("counter", "processed", "add", (5,)) in recorded.calls
    assert ("counter", "processed", "get", ()) in recorded.calls


def test_a_lock_taken_in_a_subprocess_is_acquired_and_released_at_the_parent(recorded: Recorder) -> None:
    with ipc.executor("process", reuse=True, workers=1) as pool:
        assert pool.submit(_take_the_lock).result(timeout=60) is True

    methods = [(kind, method) for kind, _, method, _ in recorded.calls]
    assert methods == [("lock", "acquire"), ("lock", "release")], "the token round-tripped both ways"
