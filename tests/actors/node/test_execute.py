"""Node task execution: dispatch, interruption, and result reconciliation."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from casty import ConnectionLostError

from skyward.actors.messages import NodeInterruptedError
from skyward.actors.node.node import Node
from skyward.infra.ssh_transport import ConnectionFailed, ConnectionRestored
from skyward.infra.worker import ResultDone, ResultPending, ResultUnknown, dumps
from skyward.infra.worker import TaskFailed as WorkerTaskFailed
from skyward.infra.worker import TaskSucceeded as WorkerTaskSucceeded

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class FakeService:
    """Answers ``run`` RPCs via a gate; optionally raises a connection error."""

    def __init__(self, owner: FakeWorker) -> None:
        self._owner = owner

    async def run(self, task_id: str, payload: bytes) -> bytes:
        owner = self._owner
        if owner.run_error is not None:
            raise owner.run_error
        if owner.execute_gate is not None:
            await owner.execute_gate.wait()
        return dumps(owner.execute_result)


class FakeControl:
    def __init__(self, owner: FakeWorker) -> None:
        self._owner = owner

    async def get_result(self, task_id: str) -> bytes:
        return dumps(self._owner.get_result_reply)


class FakeWorker:
    """Stands in for the WorkerHandle: pinned service + control proxies."""

    def __init__(self) -> None:
        self.execute_gate: asyncio.Event | None = None
        self.execute_result: Any = WorkerTaskSucceeded(result=42, node_id=0)
        self.run_error: Exception | None = None
        self.get_result_reply: Any = ResultUnknown()
        self.service = FakeService(self)
        self.control = FakeControl(self)
        self.node_id = 0


def _node(worker: FakeWorker) -> Node:
    listener = MagicMock()
    node = Node(0, listener, skip_monitor=True)
    node._worker = worker
    node._transport_up.set()
    node._active = True
    return node


async def test_execute_returns_worker_result() -> None:
    node = _node(FakeWorker())
    assert await node.execute(_forty_two, (), {}, task_id="t1") == 42


def _forty_two() -> int:
    return 42


async def test_execute_worker_failure_raises_runtime_error() -> None:
    worker = FakeWorker()
    worker.execute_result = WorkerTaskFailed(error="boom", traceback="tb", node_id=0)
    node = _node(worker)
    with pytest.raises(RuntimeError, match="boom"):
        await node.execute(_forty_two, (), {}, task_id="t1")


async def test_connection_failed_interrupts_inflight() -> None:
    worker = FakeWorker()
    worker.execute_gate = asyncio.Event()
    node = _node(worker)
    fut = asyncio.ensure_future(node.execute(_forty_two, (), {}, task_id="t1"))
    await asyncio.sleep(0.05)
    node._on_transport_event(ConnectionFailed(error="gone"))
    with pytest.raises(NodeInterruptedError):
        await fut
    node._pool.node_exhausted.assert_called_once()


async def test_dropped_rpc_reconciles_lost_reply_as_success() -> None:
    worker = FakeWorker()
    worker.run_error = ConnectionLostError("drop")
    worker.get_result_reply = ResultDone(
        result=WorkerTaskSucceeded(result="recovered", node_id=0),
    )
    node = _node(worker)
    assert await asyncio.wait_for(
        node.execute(_forty_two, (), {}, task_id="t1"), 1.0,
    ) == "recovered"


async def test_dropped_rpc_reconciles_lost_reply_as_failure() -> None:
    worker = FakeWorker()
    worker.run_error = ConnectionLostError("drop")
    worker.get_result_reply = ResultDone(
        result=WorkerTaskFailed(error="died", traceback="tb", node_id=0),
    )
    node = _node(worker)
    with pytest.raises(RuntimeError, match="died"):
        await asyncio.wait_for(node.execute(_forty_two, (), {}, task_id="t1"), 1.0)


async def test_reconcile_unknown_interrupts_for_retry() -> None:
    worker = FakeWorker()
    worker.run_error = ConnectionLostError("drop")
    worker.get_result_reply = ResultUnknown()
    node = _node(worker)
    with pytest.raises(NodeInterruptedError):
        await asyncio.wait_for(node.execute(_forty_two, (), {}, task_id="t1"), 1.0)


async def test_reconcile_pending_polls_until_done() -> None:
    worker = FakeWorker()
    worker.run_error = ConnectionLostError("drop")
    worker.get_result_reply = ResultPending()
    node = _node(worker)
    fut = asyncio.ensure_future(node.execute(_forty_two, (), {}, task_id="t1"))
    await asyncio.sleep(0.05)
    assert not fut.done()
    worker.get_result_reply = ResultDone(
        result=WorkerTaskSucceeded(result=42, node_id=0),
    )
    assert await asyncio.wait_for(fut, 5.0) == 42


async def test_dispatch_waits_for_transport_up() -> None:
    node = _node(FakeWorker())
    node._transport_up.clear()
    fut = asyncio.ensure_future(node.execute(_forty_two, (), {}, task_id="t1"))
    await asyncio.sleep(0.05)
    assert not fut.done()
    node._on_transport_event(ConnectionRestored())
    assert await asyncio.wait_for(fut, 1.0) == 42
