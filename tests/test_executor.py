"""The executor choice, from the value object down to the wire."""

import pytest

import skyward as sky
from skyward import Compute, Executor
from skyward.protocol.schemas import Worker


def test_default_is_a_reused_thread_pool_with_no_buffer() -> None:
    pool = Compute(provider=sky.Container())
    assert pool._spec.worker == Worker(concurrency=None, executor="thread", reuse=True, buffer=0)


def test_the_executor_carries_its_shape_to_the_wire() -> None:
    pool = Compute(
        provider=sky.Container(),
        executor=Executor("process", reuse=False, concurrency=4, buffer=2),
    )
    worker = pool._spec.worker
    assert (worker.executor, worker.reuse, worker.concurrency, worker.buffer) == ("process", False, 4, 2)


def test_loky_is_reusable_by_construction() -> None:
    pool = Compute(provider=sky.Container(), executor=Executor("loky"))
    assert (pool._spec.worker.executor, pool._spec.worker.reuse) == ("loky", True)


@pytest.mark.parametrize("kind", ["thread", "loky"])
def test_reuse_false_is_a_process_only_knob(kind: str) -> None:
    with pytest.raises(ValueError, match="reuse=False"):
        Executor(kind, reuse=False)  # type: ignore[arg-type]
