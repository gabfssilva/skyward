"""Values that arrive one at a time, in both directions.

A stream that hands over everything at the end is a list with extra steps, so the
tests here are about *when* an item shows up as much as what it is.
"""

import sys
import time
from collections.abc import Iterator

import cloudpickle
import pytest

import skyward as sky

pytestmark = [pytest.mark.compute, pytest.mark.xdist_group("pool")]

cloudpickle.register_pickle_by_value(sys.modules[__name__])


@sky.stream
def count(to: int) -> Iterator[int]:
    yield from range(to)


@sky.stream
def slowly(to: int) -> Iterator[float]:
    for _ in range(to):
        time.sleep(0.3)
        yield time.monotonic()


@sky.stream
def breaks() -> Iterator[str]:
    yield "before"
    raise ValueError("the generator said no")


@sky.function
def running_mean(data: Iterator[float]) -> list[float]:
    total = 0.0
    return [(total := total + value) / index for index, value in enumerate(data, 1)]


@sky.stream
def moving_average(data: Iterator[float], window: int) -> Iterator[float]:
    from collections import deque

    buffer: deque[float] = deque(maxlen=window)
    for value in data:
        buffer.append(value)
        yield sum(buffer) / len(buffer)


def describe_a_stream_out_of_a_node() -> None:
    def it_yields_in_order(pool: sky.Compute) -> None:
        assert list(count(5) >> pool) == [0, 1, 2, 3, 4]

    def it_yields_as_the_node_produces_them(pool: sky.Compute) -> None:
        arrivals = [time.monotonic() for _ in slowly(4) >> pool]
        gaps = [after - before for before, after in zip(arrivals, arrivals[1:], strict=False)]

        assert len(gaps) == 3
        assert min(gaps) > 0.25, f"the items arrived together, so they came from a buffer: {gaps}"

    def describe_when_the_consumer_walks_away() -> None:
        def it_leaves_the_worker_free_for_the_next_call(pool: sky.Compute) -> None:
            """A million items were asked for and one was read; the rest are nobody's."""
            abandoned = count(1_000_000) >> pool

            assert next(abandoned) == 0
            del abandoned

            assert list(count(3) >> pool) == [0, 1, 2]

    def describe_when_the_generator_raises_partway() -> None:
        def it_delivers_what_came_before_and_then_the_failure(pool: sky.Compute) -> None:
            items = breaks() >> pool

            assert next(items) == "before"

            with pytest.raises(sky.TaskFailedError) as raised:
                next(items)

            assert "the generator said no" in raised.value.message


def describe_a_stream_into_a_node() -> None:
    def it_feeds_an_iterator_argument_as_it_is_consumed(pool: sky.Compute) -> None:
        assert running_mean(iter([1.0, 2.0, 3.0, 4.0])) >> pool == [1.0, 1.5, 2.0, 2.5]

    def it_streams_both_ways_at_once(pool: sky.Compute) -> None:
        averages = moving_average(iter([1.0, 2.0, 3.0, 4.0]), window=2) >> pool

        assert list(averages) == [1.0, 1.5, 2.5, 3.5]
