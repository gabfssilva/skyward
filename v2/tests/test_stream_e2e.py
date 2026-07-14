"""A generator, on a machine, read one item at a time.

The claim being tested is not that the items arrive — it is that they arrive as
they are produced, and that a failure halfway through is still a failure the caller
sees, after the items it already had.
"""

import sys
import time
from collections.abc import Iterator
from pathlib import Path

import cloudpickle
import pytest

import skyward2 as skyward
from skyward2 import Compute, TaskFailedError

pytestmark = pytest.mark.e2e

IMAGE = skyward.Image(python="3.13", skyward="local")

cloudpickle.register_pickle_by_value(sys.modules[__name__])


@skyward.stream
def count(to: int) -> Iterator[int]:
    yield from range(to)


@skyward.stream
def slowly(to: int) -> Iterator[float]:
    for _ in range(to):
        time.sleep(0.3)
        yield time.monotonic()


@skyward.stream
def breaks() -> Iterator[str]:
    yield "before"
    raise ValueError("the generator said no")


@pytest.fixture
def pool(tmp_path: Path):
    with skyward.Compute(
        provider=skyward.Container(),
        nodes=1,
        cpus=1,
        memory_gb=1,
        image=IMAGE,
        database=tmp_path / "skyward.sqlite",
    ) as pool:
        yield pool


def test_the_items_come_back_in_order(pool: Compute):
    assert list(count(5) >> pool) == [0, 1, 2, 3, 4]


def test_the_items_arrive_as_they_are_produced(pool: Compute):
    """Otherwise this is not a stream, it is a list with extra steps.

    The generator sleeps between items, so a caller that gets them all at once got
    them from a buffer. The arrivals are spaced because the production is.
    """
    arrivals = [time.monotonic() for _ in slowly(4) >> pool]
    gaps = [after - before for before, after in zip(arrivals, arrivals[1:], strict=False)]

    assert len(gaps) == 3
    assert min(gaps) > 0.25, f"the items arrived together: {gaps}"


def test_a_consumer_that_stops_reading_stops_the_generator(pool: Compute):
    items = count(1_000_000) >> pool

    assert next(items) == 0
    items.close()


def test_a_failure_arrives_after_the_items_that_came_before_it(pool: Compute):
    items = breaks() >> pool

    assert next(items) == "before"

    with pytest.raises(TaskFailedError) as raised:
        next(items)

    assert "the generator said no" in raised.value.message
    assert "ValueError" in (raised.value.details.get("traceback") or "")


def test_a_generator_decorated_as_a_function_is_refused_where_it_is_written():
    with pytest.raises(TypeError, match="@stream"):

        @skyward.function
        def wrong() -> Iterator[int]:
            yield 1
