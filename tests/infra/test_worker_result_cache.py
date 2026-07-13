"""Tests for the worker's result cache + get_result reconciliation path.

Validates the contract used by the node on a dropped ``run`` RPC:
after a task is dispatched with a non-empty ``task_id``, the worker can
later answer ``WorkerControl.get_result(task_id)`` with one of:

- ``ResultPending`` — task accepted, still running
- ``ResultDone(result)`` — task completed; payload is the original outcome
- ``ResultUnknown`` — never seen, evicted by TTL/size, or worker restarted
"""
from __future__ import annotations

import asyncio

import casty
import pytest

import skyward.infra.worker as worker_mod
from skyward.infra.worker import (
    ResultDone,
    ResultPending,
    ResultUnknown,
    StreamStarted,
    TaskFailed,
    TaskSucceeded,
    WorkerControl,
    WorkerService,
    _Runtime,
    dumps,
    loads,
)

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


@pytest.fixture
async def system():
    async with casty.local() as s:
        worker_mod._runtime = _Runtime(
            node_id=0, loop=asyncio.get_running_loop(),
        )
        yield s
        worker_mod._runtime = None


def _payload(fn, args=(), kwargs=None) -> bytes:
    return dumps((fn, args, kwargs or {}, ()))


async def _run(system, task_id: str, fn) -> object:
    service = system.service(WorkerService)
    return loads(await service.run(task_id, _payload(fn)))


async def _get_result(system, task_id: str) -> object:
    control = system.service(WorkerControl)
    return loads(await control.get_result(task_id))


class TestResultCache:
    async def test_pending_then_done(self, system, tmp_path) -> None:
        """A still-running task reports Pending; once finished, Done."""
        gate_file = str(tmp_path / "gate")

        def slow_task(path: str = gate_file) -> int:
            import os
            import time

            deadline = time.monotonic() + 5.0
            while not os.path.exists(path) and time.monotonic() < deadline:
                time.sleep(0.01)
            return 42

        run_task = asyncio.ensure_future(_run(system, "t1", slow_task))
        await asyncio.sleep(0.1)

        assert isinstance(await _get_result(system, "t1"), ResultPending)

        (tmp_path / "gate").touch()
        result = await asyncio.wait_for(run_task, timeout=2.0)
        assert isinstance(result, TaskSucceeded)
        assert result.result == 42

        done_reply = await _get_result(system, "t1")
        assert isinstance(done_reply, ResultDone)
        assert isinstance(done_reply.result, TaskSucceeded)
        assert done_reply.result.result == 42

    async def test_unknown_task_id(self, system) -> None:
        """Asking for a task_id never dispatched returns ResultUnknown."""
        assert isinstance(await _get_result(system, "never-seen"), ResultUnknown)

    async def test_failed_task_cached(self, system) -> None:
        """A task that raises is cached as ResultDone(TaskFailed)."""

        def boom() -> int:
            raise RuntimeError("bang")

        result = await _run(system, "t-fail", boom)
        assert isinstance(result, TaskFailed)

        reply = await _get_result(system, "t-fail")
        assert isinstance(reply, ResultDone)
        assert isinstance(reply.result, TaskFailed)
        assert "bang" in reply.result.error

    async def test_ttl_eviction(self, system) -> None:
        """Entries past TTL are evicted on the next cache read."""
        worker_mod._runtime.result_cache_ttl = 0.05

        await _run(system, "t-ttl", lambda: 1)
        await asyncio.sleep(0.15)
        assert isinstance(await _get_result(system, "t-ttl"), ResultUnknown)

    async def test_stream_not_cached(self, system) -> None:
        """Generator results are live pull streams and must not be cached."""

        def gen():
            yield 1
            yield 2

        result = await _run(system, "t-stream", gen)
        assert isinstance(result, StreamStarted)

        assert isinstance(await _get_result(system, "t-stream"), ResultUnknown)

    async def test_no_task_id_skips_cache(self, system) -> None:
        """A task dispatched with empty task_id leaves no cache entry."""
        result = await _run(system, "", lambda: 7)
        assert isinstance(result, TaskSucceeded)
        assert isinstance(await _get_result(system, ""), ResultUnknown)
