"""Sending work to a pool — every operator the README promises, on real machines.

Nothing below knows there is a daemon, a database or an SSH connection. What is
written here is what a user writes, which is the only reason a green run means
anything: the failure these catch is the failure a user would have hit.
"""

import functools
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from contextvars import Context

import cloudpickle
import msgspec
import pytest

import skyward as sky
from skyward.shared.schemas import FunctionExcerpt

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
def tripled(x: int) -> int:
    return x * 3


def scaled(factor: int, x: int) -> int:
    return factor * x


@sky.function(retry=None)
def crash() -> None:
    os.kill(os.getpid(), signal.SIGSEGV)


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

    def describe_when_it_takes_the_worker_down_with_it() -> None:
        def it_is_lost_at_once_and_the_worker_comes_back(pool: sky.Compute) -> None:
            """On the thread executor a segfault ends the worker's own process; what is left to check is that nobody waits on it."""
            started = time.monotonic()

            with pytest.raises(sky.TaskIndeterminateError):
                crash() >> pool

            assert time.monotonic() - started < 60, "the attempt was declared lost when the worker came back, not when a link next dropped"
            assert sorted(rank for _, rank in where_am_i() @ pool) == [0, 1], "and the restarted worker takes work again"

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

    def it_sends_the_text_of_the_function_beside_it(pool: sky.Compute, monkeypatch: pytest.MonkeyPatch) -> None:
        """A pickle is compiled code with no text in it, so what the console shows is read off this file and sent along."""
        client = pool.client
        original = client.call
        sent: list[str] = []

        async def spying[T](
            method: str,
            path: str,
            kind: type[T],
            /,
            body: bytes | None = None,
            headers: dict[str, str] | None = None,
            urgent: bool = False,
            **query: object,
        ) -> T:
            if path.endswith("/excerpt") and body is not None:
                sent.append(msgspec.json.decode(body, type=FunctionExcerpt).text)
            return await original(method, path, kind, body=body, headers=headers, urgent=urgent, **query)

        monkeypatch.setattr(client, "call", spying)

        assert tripled(3) >> pool == 9
        assert sent == ["import skyward as sky\n\n\n@sky.function\ndef tripled(x: int) -> int:\n    return x * 3\n"]

    def a_partial_goes_by_the_name_and_the_text_of_the_function_it_was_made_of(pool: sky.Compute, monkeypatch: pytest.MonkeyPatch) -> None:
        """A partial has no name and no file of its own; the function inside it has both."""
        client = pool.client
        uploading, calling = client.upload, client.call
        named: list[str] = []
        sent: list[str] = []

        async def upload(path: str, body: bytes, headers: dict[str, str] | None = None) -> None:
            if path.startswith("/v1/functions/") and headers is not None:
                named.append(headers["X-Skyward-Function-Name"])
            await uploading(path, body, headers)

        async def call[T](
            method: str,
            path: str,
            kind: type[T],
            /,
            body: bytes | None = None,
            headers: dict[str, str] | None = None,
            urgent: bool = False,
            **query: object,
        ) -> T:
            if path.endswith("/excerpt") and body is not None:
                sent.append(msgspec.json.decode(body, type=FunctionExcerpt).text)
            return await calling(method, path, kind, body=body, headers=headers, urgent=urgent, **query)

        monkeypatch.setattr(client, "upload", upload)
        monkeypatch.setattr(client, "call", call)

        assert sky.function(functools.partial(scaled, 4))(3) >> pool == 12
        assert named == ["scaled"]
        assert sent == ["def scaled(factor: int, x: int) -> int:\n    return factor * x\n"]


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
