"""Values that arrive one at a time, in both directions.

A stream that hands over everything at the end is a list with extra steps, so the
tests here are about *when* an item shows up as much as what it is.
"""

import asyncio
import logging
import sys
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import ClassVar

import casty
import cloudpickle
import msgspec
import pytest

import skyward as sky
from skyward.shared import codec
from skyward.shared.frames import Chunk, End, Step
from skyward.shared.observability.logger import NAME
from skyward.worker import worker
from skyward.worker.api import Info
from skyward.worker.plugins import Plugin

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


@pytest.mark.compute
@pytest.mark.xdist_group("pool")
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


@pytest.mark.compute
@pytest.mark.xdist_group("pool")
def describe_a_stream_into_a_node() -> None:
    def it_feeds_an_iterator_argument_as_it_is_consumed(pool: sky.Compute) -> None:
        assert running_mean(iter([1.0, 2.0, 3.0, 4.0])) >> pool == [1.0, 1.5, 2.0, 2.5]

    def it_streams_both_ways_at_once(pool: sky.Compute) -> None:
        averages = moving_average(iter([1.0, 2.0, 3.0, 4.0]), window=2) >> pool

        assert list(averages) == [1.0, 1.5, 2.5, 3.5]


hook_entered = threading.Event()
hook_released = threading.Event()
hook_returned = threading.Event()


class Holding(Plugin, frozen=True):
    """A plugin whose ``run`` blocks until the test lets it go."""

    kind: ClassVar[str] = "holding"

    def run[T](self, call: Callable[[], T], info: Info) -> T:
        hook_entered.set()
        hook_released.wait(3)
        hook_returned.set()
        return call()


class Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.mark.local
def describe_the_worker_s_stream_lifecycle() -> None:
    @pytest.fixture
    def on_a_node(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
        monkeypatch.setenv("SKYWARD_NODE", "nod_test")
        monkeypatch.setenv("SKYWARD_COMPUTE", "cmp_test")
        monkeypatch.setenv("SKYWARD_RANK", "0")
        monkeypatch.setenv("SKYWARD_PEERS", "10.0.0.1")
        monkeypatch.setenv("SKYWARD_PLUGINS", "[]")
        monkeypatch.setattr(worker, "MODE", "thread")
        with ThreadPoolExecutor(2) as pool:
            monkeypatch.setattr(worker, "thread_pool", pool, raising=False)
            yield
        worker.generators.clear()
        worker.pulling.clear()

    async def it_closes_an_abandoned_stream_whose_cleanup_blocks_on_the_loop(on_a_node: None) -> None:
        loop = asyncio.get_running_loop()
        cleaned = threading.Event()

        async def acquire() -> str:
            return "held"

        def guarded() -> Iterator[int]:
            try:
                yield 1
                yield 2
            finally:
                assert asyncio.run_coroutine_threadsafe(acquire(), loop).result(timeout=2) == "held"
                cleaned.set()

        stream = guarded()
        assert next(stream) == 1
        worker.generators["exe_abandoned"] = stream

        system = casty.local()
        try:
            async with asyncio.timeout(5):
                await system.service(worker.Worker).close("exe_abandoned")
        finally:
            await system.close()

        assert cleaned.is_set(), "the cleanup reached the loop and came back, so it did not run on the loop"
        assert "exe_abandoned" not in worker.generators

    async def it_lets_a_running_pull_finish_before_finalizing_the_generator(on_a_node: None) -> None:
        release = threading.Event()
        order: list[str] = []

        def slow() -> Iterator[int]:
            try:
                yield 1
                release.wait(5)
                order.append("pulled")
                yield 2
            finally:
                order.append("finalized")

        stream = slow()
        assert next(stream) == 1
        worker.generators["exe_mid_pull"] = stream

        system = casty.local()
        try:
            stepping = asyncio.create_task(worker.advance("exe_mid_pull"))
            async with asyncio.timeout(5):
                while "exe_mid_pull" not in worker.pulling:
                    await asyncio.sleep(0.01)
                await system.service(worker.Worker).close("exe_mid_pull")

            assert order == [], "a generator that is executing is not closed under the pull"
            assert not stepping.done()

            release.set()
            async with asyncio.timeout(5):
                step = await stepping
        finally:
            release.set()
            await system.close()

        assert step == End(), "a stream closed mid-pull ends instead of delivering to nobody"
        assert order == ["pulled", "finalized"]
        assert "exe_mid_pull" not in worker.generators

    async def it_logs_a_cleanup_that_raises_and_closes_normally(on_a_node: None) -> None:
        def failing() -> Iterator[int]:
            try:
                yield 1
                yield 2
            finally:
                raise RuntimeError("the cleanup broke")

        stream = failing()
        assert next(stream) == 1
        worker.generators["exe_broken_cleanup"] = stream

        records = Records()
        target = logging.getLogger(NAME)
        previous = target.level
        target.addHandler(records)
        target.setLevel(logging.WARNING)
        system = casty.local()
        try:
            async with asyncio.timeout(5):
                await system.service(worker.Worker).close("exe_broken_cleanup")
        finally:
            await system.close()
            target.removeHandler(records)
            target.setLevel(previous)

        assert any(
            record.exc_info is not None and isinstance(record.exc_info[1], RuntimeError) and "the cleanup broke" in str(record.exc_info[1])
            for record in records.records
        ), [record.getMessage() for record in records.records]

    async def it_answers_a_ping_while_a_plugin_s_run_hook_blocks_the_open(on_a_node: None, monkeypatch: pytest.MonkeyPatch) -> None:
        hook_entered.clear()
        hook_released.clear()
        hook_returned.clear()
        monkeypatch.setattr(worker, "installed", (Holding(),))

        def numbers() -> Iterator[int]:
            yield 7

        system = casty.local()
        try:
            opening = asyncio.create_task(system.service(worker.Worker).open("exe_opened", codec.dumps(numbers), codec.dumps(((), {}))))
            async with asyncio.timeout(5):
                while not hook_entered.is_set():
                    await asyncio.sleep(0.01)
                node = await system.service(worker.Control).ping()
            answered_while_blocked = not hook_returned.is_set()

            hook_released.set()
            async with asyncio.timeout(5):
                await opening
                first = msgspec.msgpack.decode(await system.service(worker.Worker).step("exe_opened"), type=Step)
        finally:
            hook_released.set()
            await system.close()

        assert node == "nod_test"
        assert answered_while_blocked, "the ping was answered only after the plugin's hook had returned"
        match first:
            case Chunk(value=value):
                assert codec.loads(value) == 7
            case _:
                pytest.fail(f"the opened stream yielded nothing: {first}")
