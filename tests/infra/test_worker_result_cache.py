"""Tests for the worker's result cache + GetResult reconciliation path.

Validates the contract used by the node actor on ``ConnectionRestored``:
after a task is dispatched with a non-empty ``task_id``, the worker can
later answer ``GetResult(task_id)`` with one of:

- ``ResultPending`` — task accepted, still running
- ``ResultDone(result)`` — task completed; payload is the original outcome
- ``ResultUnknown`` — never seen, evicted by TTL/size, or worker restarted
"""
from __future__ import annotations

import asyncio
import threading

import pytest
from casty import ActorRef, ActorSystem

from skyward.infra.streaming import _StreamHandle
from skyward.infra.worker import (
    ExecuteTask,
    GetResult,
    GetResultReply,
    ResultDone,
    ResultPending,
    ResultUnknown,
    TaskFailed,
    TaskResult,
    TaskSucceeded,
    worker_behavior,
)


pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


@pytest.fixture
async def system():
    s = ActorSystem("test-worker-cache")
    yield s
    await s.shutdown()


def _capture(loop: asyncio.AbstractEventLoop) -> tuple[ActorRef, asyncio.Future]:
    """Return a one-shot reply ref and its result future.

    The ref is a thin shim that fulfils the future on first ``tell``.
    """
    fut: asyncio.Future = loop.create_future()

    class _Ref:
        def tell(self, msg: object) -> None:
            if not fut.done():
                fut.set_result(msg)

    return _Ref(), fut  # type: ignore[return-value]


async def _ask_get_result(
    system: ActorSystem,
    worker_ref: ActorRef,
    task_id: str,
    timeout: float = 2.0,
) -> GetResultReply:
    """Send ``GetResult`` and await the reply."""
    loop = asyncio.get_running_loop()
    reply_ref, fut = _capture(loop)
    worker_ref.tell(GetResult(task_id=task_id, reply_to=reply_ref))
    return await asyncio.wait_for(fut, timeout=timeout)


class TestResultCache:
    @pytest.mark.asyncio
    async def test_pending_then_done(self, system: ActorSystem) -> None:
        """A still-running task reports Pending; once finished, Done."""
        worker = system.spawn(worker_behavior(node_id=0), "worker")

        gate = threading.Event()

        def slow_task() -> int:
            # block on a real threading event — the task runs in to_thread
            gate.wait(timeout=5.0)
            return 42

        loop = asyncio.get_running_loop()
        reply_ref, exec_fut = _capture(loop)
        worker.tell(ExecuteTask(
            fn=slow_task, args=(), kwargs={},
            reply_to=reply_ref, task_id="t1",
        ))

        # give the actor a tick to register "running"
        await asyncio.sleep(0.05)

        pending_reply = await _ask_get_result(system, worker, "t1")
        assert isinstance(pending_reply, ResultPending)

        # release the task
        gate.set()
        result: TaskResult = await asyncio.wait_for(exec_fut, timeout=2.0)
        assert isinstance(result, TaskSucceeded)
        assert result.result == 42

        done_reply = await _ask_get_result(system, worker, "t1")
        assert isinstance(done_reply, ResultDone)
        assert isinstance(done_reply.result, TaskSucceeded)
        assert done_reply.result.result == 42

    @pytest.mark.asyncio
    async def test_unknown_task_id(self, system: ActorSystem) -> None:
        """Asking for a task_id never dispatched returns ResultUnknown."""
        worker = system.spawn(worker_behavior(node_id=0), "worker")
        reply = await _ask_get_result(system, worker, "never-seen")
        assert isinstance(reply, ResultUnknown)

    @pytest.mark.asyncio
    async def test_failed_task_cached(self, system: ActorSystem) -> None:
        """A task that raises is cached as ResultDone(TaskFailed)."""
        worker = system.spawn(worker_behavior(node_id=0), "worker")

        def boom() -> int:
            raise RuntimeError("bang")

        loop = asyncio.get_running_loop()
        reply_ref, exec_fut = _capture(loop)
        worker.tell(ExecuteTask(
            fn=boom, args=(), kwargs={},
            reply_to=reply_ref, task_id="t-fail",
        ))
        result = await asyncio.wait_for(exec_fut, timeout=2.0)
        assert isinstance(result, TaskFailed)

        reply = await _ask_get_result(system, worker, "t-fail")
        assert isinstance(reply, ResultDone)
        assert isinstance(reply.result, TaskFailed)
        assert "bang" in reply.result.error

    @pytest.mark.asyncio
    async def test_ttl_eviction(self, system: ActorSystem) -> None:
        """Entries past TTL are evicted on the next cache read."""
        worker = system.spawn(
            worker_behavior(node_id=0, result_cache_ttl=0.05),
            "worker-short-ttl",
        )

        loop = asyncio.get_running_loop()
        reply_ref, exec_fut = _capture(loop)
        worker.tell(ExecuteTask(
            fn=lambda: 1, args=(), kwargs={},
            reply_to=reply_ref, task_id="t-ttl",
        ))
        await asyncio.wait_for(exec_fut, timeout=2.0)

        await asyncio.sleep(0.15)
        reply = await _ask_get_result(system, worker, "t-ttl")
        assert isinstance(reply, ResultUnknown)

    @pytest.mark.asyncio
    async def test_streamhandle_not_cached(self, system: ActorSystem) -> None:
        """Generator results carry a live producer_ref and must not be cached."""
        worker = system.spawn(worker_behavior(node_id=0, system=system), "worker-gen")

        def gen():
            yield 1
            yield 2

        loop = asyncio.get_running_loop()
        reply_ref, exec_fut = _capture(loop)
        worker.tell(ExecuteTask(
            fn=gen, args=(), kwargs={},
            reply_to=reply_ref, task_id="t-stream",
        ))
        result = await asyncio.wait_for(exec_fut, timeout=2.0)
        assert isinstance(result, TaskSucceeded)
        assert isinstance(result.result, _StreamHandle)

        reply = await _ask_get_result(system, worker, "t-stream")
        assert isinstance(reply, ResultUnknown)

    @pytest.mark.asyncio
    async def test_no_task_id_skips_cache(self, system: ActorSystem) -> None:
        """A task dispatched with empty task_id leaves no cache entry."""
        worker = system.spawn(worker_behavior(node_id=0), "worker-notid")

        loop = asyncio.get_running_loop()
        reply_ref, exec_fut = _capture(loop)
        worker.tell(ExecuteTask(
            fn=lambda: 7, args=(), kwargs={},
            reply_to=reply_ref, task_id="",
        ))
        await asyncio.wait_for(exec_fut, timeout=2.0)

        reply = await _ask_get_result(system, worker, "")
        assert isinstance(reply, ResultUnknown)


