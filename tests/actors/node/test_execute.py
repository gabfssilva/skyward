"""Node task execution: dispatch, interruption, and result reconciliation."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from skyward.actors.messages import NodeInterruptedError
from skyward.actors.node.node import Node
from skyward.infra.ssh_transport import ConnectionFailed, ConnectionLost, ConnectionRestored
from skyward.infra.worker import GetResult, ResultDone, ResultPending, ResultUnknown
from skyward.infra.worker import TaskFailed as WorkerTaskFailed
from skyward.infra.worker import TaskSucceeded as WorkerTaskSucceeded

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class FakeClient:
    """Answers worker asks: ExecuteTask via a gate, GetResult via a reply."""

    def __init__(self) -> None:
        self.execute_gate: asyncio.Event | None = None
        self.execute_result: Any = WorkerTaskSucceeded(result=42, node_id=0)
        self.get_result_reply: Any = ResultUnknown()

    async def ask(self, _ref: Any, factory: Any, timeout: float = 10.0) -> Any:
        msg = factory(MagicMock())
        if isinstance(msg, GetResult):
            return self.get_result_reply
        if self.execute_gate is not None:
            await self.execute_gate.wait()
        return self.execute_result


def _node(client: FakeClient) -> Node:
    listener = MagicMock()
    node = Node(0, listener, skip_monitor=True)
    node._client = client
    node._worker_ref = MagicMock()
    node._transport_up.set()
    node._active = True
    return node


async def test_execute_returns_worker_result() -> None:
    client = FakeClient()
    node = _node(client)
    assert await node.execute(b"fn", (), {}, task_id="t1") == 42


async def test_execute_worker_failure_raises_runtime_error() -> None:
    client = FakeClient()
    client.execute_result = WorkerTaskFailed(error="boom", traceback="tb", node_id=0)
    node = _node(client)
    with pytest.raises(RuntimeError, match="boom"):
        await node.execute(b"fn", (), {}, task_id="t1")


async def test_connection_failed_interrupts_inflight() -> None:
    client = FakeClient()
    client.execute_gate = asyncio.Event()
    node = _node(client)
    fut = asyncio.ensure_future(node.execute(b"fn", (), {}, task_id="t1"))
    await asyncio.sleep(0.05)
    node._on_transport_event(ConnectionFailed(error="gone"))
    with pytest.raises(NodeInterruptedError):
        await fut
    node._pool.node_exhausted.assert_called_once()


async def test_reconnect_reconciles_lost_reply_as_success() -> None:
    client = FakeClient()
    client.execute_gate = asyncio.Event()  # original ask never resolves
    client.get_result_reply = ResultDone(
        result=WorkerTaskSucceeded(result="recovered", node_id=0),
    )
    node = _node(client)
    fut = asyncio.ensure_future(node.execute(b"fn", (), {}, task_id="t1"))
    await asyncio.sleep(0.05)
    node._on_transport_event(ConnectionLost(error="drop"))
    node._on_transport_event(ConnectionRestored())
    assert await asyncio.wait_for(fut, 1.0) == "recovered"


async def test_reconnect_reconciles_lost_reply_as_failure() -> None:
    client = FakeClient()
    client.execute_gate = asyncio.Event()
    client.get_result_reply = ResultDone(
        result=WorkerTaskFailed(error="died", traceback="tb", node_id=0),
    )
    node = _node(client)
    fut = asyncio.ensure_future(node.execute(b"fn", (), {}, task_id="t1"))
    await asyncio.sleep(0.05)
    node._on_transport_event(ConnectionRestored())
    with pytest.raises(RuntimeError, match="died"):
        await asyncio.wait_for(fut, 1.0)


async def test_reconcile_unknown_interrupts_for_retry() -> None:
    client = FakeClient()
    client.execute_gate = asyncio.Event()
    client.get_result_reply = ResultUnknown()
    node = _node(client)
    fut = asyncio.ensure_future(node.execute(b"fn", (), {}, task_id="t1"))
    await asyncio.sleep(0.05)
    node._on_transport_event(ConnectionRestored())
    with pytest.raises(NodeInterruptedError):
        await asyncio.wait_for(fut, 1.0)


async def test_reconcile_pending_waits_for_original_ask() -> None:
    client = FakeClient()
    client.execute_gate = asyncio.Event()
    client.get_result_reply = ResultPending()
    node = _node(client)
    fut = asyncio.ensure_future(node.execute(b"fn", (), {}, task_id="t1"))
    await asyncio.sleep(0.05)
    node._on_transport_event(ConnectionRestored())
    await asyncio.sleep(0.05)
    assert not fut.done()
    client.execute_gate.set()
    assert await asyncio.wait_for(fut, 1.0) == 42


async def test_dispatch_waits_for_transport_up() -> None:
    client = FakeClient()
    node = _node(client)
    node._transport_up.clear()
    fut = asyncio.ensure_future(node.execute(b"fn", (), {}, task_id="t1"))
    await asyncio.sleep(0.05)
    assert not fut.done()
    node._on_transport_event(ConnectionRestored())
    assert await asyncio.wait_for(fut, 1.0) == 42
