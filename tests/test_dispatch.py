"""Sending work to a pool — every operator the README promises, on real machines.

Nothing below knows there is a daemon, a database or an SSH connection. What is
written here is what a user writes, which is the only reason a green run means
anything: the failure these catch is the failure a user would have hit.
"""

import os
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from contextvars import Context

import cloudpickle
import pytest

import skyward as sky

pytestmark = [pytest.mark.compute, pytest.mark.xdist_group("pool")]

cloudpickle.register_pickle_by_value(sys.modules[__name__])
"""Ship this module's functions by value — a test module is not importable on a node."""


@sky.function
def double(x: int) -> int:
    return x * 2


@sky.function
def slow(seconds: float) -> float:
    time.sleep(seconds)
    return seconds


@sky.function
def blow_up() -> int:
    raise ValueError("the function said no")


@sky.function
def echo_reversed(payload: bytes) -> bytes:
    return payload[::-1]


@sky.function
def counted(x: int) -> int:
    return x + 1


@sky.function
def where_am_i() -> tuple[str, int]:
    info = sky.instance_info()
    return info.node, info.rank


def describe_dispatching_a_call() -> None:
    def it_runs_on_one_node_and_gives_the_value_back(pool: sky.Compute) -> None:
        assert double(21) >> pool == 42

    def it_broadcasts_to_every_node(pool: sky.Compute) -> None:
        seen = where_am_i() @ pool

        assert sorted(rank for _, rank in seen) == [0, 1]
        assert len({node for node, _ in seen}) == 2, "two nodes, two answers"

    def it_runs_a_group_in_parallel(pool: sky.Compute) -> None:
        started = time.monotonic()

        assert (slow(2.0) & slow(2.0) & slow(2.0)) >> pool == [2.0, 2.0, 2.0]
        assert time.monotonic() - started < 5.0, "three two-second calls did not run one after another"

    def it_gathers_the_same_way_the_ampersand_does(pool: sky.Compute) -> None:
        assert sky.gather(double(1), double(2)) >> pool == [2, 4]

    def it_keeps_the_order_of_a_map(pool: sky.Compute) -> None:
        assert pool.map(double, range(6)) == [0, 2, 4, 6, 8, 10]

    def describe_asynchronously() -> None:
        def it_hands_back_a_future_that_does_not_block(pool: sky.Compute) -> None:
            future = slow(3.0) > pool

            assert double(2) >> pool == 4, "the pool kept working while the future was in flight"
            assert future.result() == 3.0

    def describe_when_the_function_raises() -> None:
        def it_arrives_as_a_failure_carrying_the_remote_traceback(pool: sky.Compute) -> None:
            with pytest.raises(sky.TaskFailedError) as raised:
                _ = blow_up() >> pool

            assert "the function said no" in raised.value.message
            assert "ValueError" in (raised.value.details.get("traceback") or ""), "the remote traceback survives the trip"

    def describe_when_it_outlives_its_timeout() -> None:
        def it_fails_instead_of_hanging_and_leaves_the_node_usable(pool: sky.Compute) -> None:
            with pytest.raises(sky.SkywardError):
                _ = slow(60).with_timeout(3) >> pool

            assert double(1) >> pool == 2


def describe_the_implicit_pool() -> None:
    def it_stands_in_for_the_pool_the_block_opened(pool: sky.Compute) -> None:
        assert double(4) >> sky.sky == 8, "the open block is the session's pool"

    def it_says_so_when_there_is_no_block_open() -> None:
        """A fresh context is the only place with no block open — the suite keeps one up."""

        def outside() -> None:
            with pytest.raises(RuntimeError, match="no pool to run on"):
                _ = double(4) >> sky.sky

        Context().run(outside)


MEBIBYTE = 1024 * 1024


def describe_payloads_over_four_mebibytes() -> None:
    def it_carries_eight_mebibytes_to_the_worker_and_eight_back(pool: sky.Compute) -> None:
        payload = os.urandom(8 * MEBIBYTE)

        returned = echo_reversed(payload) >> pool

        assert len(returned) == 8 * MEBIBYTE
        assert returned == payload[::-1]


def describe_function_uploads() -> None:
    def it_uploads_a_function_once_and_sends_only_its_digest_after(pool: sky.Compute, monkeypatch: pytest.MonkeyPatch) -> None:
        client = pool.client
        original = client.upload
        uploads: list[str] = []

        async def counting(path: str, body: bytes, headers: dict[str, str] | None = None) -> None:
            uploads.append(path)
            await original(path, body, headers)

        monkeypatch.setattr(client, "upload", counting)

        assert [counted(n) >> pool for n in range(3)] == [1, 2, 3]
        assert len([path for path in uploads if path.startswith("/v1/functions/")]) == 1


def describe_callbacks_on_an_async_future() -> None:
    def _settled[T](future: Future[T], callback: Callable[[Future[T]], None]) -> threading.Event:
        done = threading.Event()

        def wrapped(settled: Future[T]) -> None:
            try:
                callback(settled)
            finally:
                done.set()

        future.add_done_callback(wrapped)
        return done

    def it_runs_them_on_the_callbacks_thread_not_the_loop(pool: sky.Compute) -> None:
        names: list[str] = []
        future = double(5) > pool

        done = _settled(future, lambda _: names.append(threading.current_thread().name))

        assert future.result() == 10
        assert done.wait(30), "the callback never ran"
        assert names == ["skyward-callbacks"]

    def it_lets_a_callback_dispatch_synchronously_without_deadlocking(pool: sky.Compute) -> None:
        results: list[int] = []
        future = double(1) > pool

        done = _settled(future, lambda settled: results.append(double(settled.result()) >> pool))

        assert done.wait(60), "a callback dispatching with >> deadlocked"
        assert results == [4]
