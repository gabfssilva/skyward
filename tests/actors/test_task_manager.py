"""TaskManager: round-robin, per-node slots, retry, targeting, broadcast."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from skyward.actors.messages import NodeInterruptedError, PressureReport
from skyward.actors.task_manager import TaskManager

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class FakeNode:
    def __init__(self, node_id: int, *, result: Any = "ok") -> None:
        self.node_id = node_id
        self.result = result
        self.executed: list[str] = []
        self.gate: asyncio.Event | None = None
        self.fail_times = 0

    async def execute(
        self, fn: Any, args: tuple, kwargs: dict, *, timeout: float, task_id: str,
    ) -> Any:
        self.executed.append(task_id)
        if self.gate is not None:
            await self.gate.wait()
        if self.fail_times > 0:
            self.fail_times -= 1
            raise NodeInterruptedError(self.node_id, "preempted")
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


async def test_submit_dispatches_and_returns_value() -> None:
    tm = TaskManager()
    node = FakeNode(0)
    tm.node_available(0, node, slots=2)
    result = await tm.submit(b"fn", (), {}, task_id="t1")
    assert result == "ok"
    assert node.executed == ["t1"]


async def test_round_robin_across_nodes() -> None:
    tm = TaskManager()
    n0, n1 = FakeNode(0), FakeNode(1)
    tm.node_available(0, n0, slots=4)
    tm.node_available(1, n1, slots=4)
    await asyncio.gather(*(
        tm.submit(b"fn", (), {}, task_id=f"t{i}") for i in range(4)
    ))
    assert len(n0.executed) == 2
    assert len(n1.executed) == 2


async def test_queues_when_no_free_slot_and_drains() -> None:
    tm = TaskManager()
    node = FakeNode(0)
    node.gate = asyncio.Event()
    tm.node_available(0, node, slots=1)

    first = asyncio.ensure_future(tm.submit(b"fn", (), {}, task_id="t1"))
    second = asyncio.ensure_future(tm.submit(b"fn", (), {}, task_id="t2"))
    await asyncio.sleep(0.05)
    assert node.executed == ["t1"]

    node.gate.set()
    assert await first == "ok"
    assert await second == "ok"
    assert node.executed == ["t1", "t2"]


async def test_user_error_propagates_without_retry() -> None:
    tm = TaskManager(retry_on_interruption=3)
    node = FakeNode(0, result=ValueError("boom"))
    tm.node_available(0, node, slots=1)
    with pytest.raises(ValueError, match="boom"):
        await tm.submit(b"fn", (), {}, task_id="t1")
    assert node.executed == ["t1"]


async def test_interrupted_task_retries_then_succeeds() -> None:
    tm = TaskManager(retry_on_interruption=3)
    node = FakeNode(0)
    node.fail_times = 2
    tm.node_available(0, node, slots=1)
    result = await tm.submit(b"fn", (), {}, task_id="t1")
    assert result == "ok"
    assert node.executed == ["t1", "t1", "t1"]


async def test_interrupted_task_exhausts_retries() -> None:
    tm = TaskManager(retry_on_interruption=2)
    node = FakeNode(0)
    node.fail_times = 10
    tm.node_available(0, node, slots=1)
    with pytest.raises(NodeInterruptedError):
        await tm.submit(b"fn", (), {}, task_id="t1")
    assert len(node.executed) == 3  # first + 2 retries


async def test_target_routes_to_specific_rank() -> None:
    tm = TaskManager()
    n0, n1 = FakeNode(0), FakeNode(1)
    tm.node_available(0, n0, slots=1)
    tm.node_available(1, n1, slots=1)
    await tm.submit(b"fn", (), {}, task_id="t1", target=1)
    assert n1.executed == ["t1"]
    assert n0.executed == []


async def test_target_head_routes_to_rank_zero() -> None:
    tm = TaskManager()
    n0 = FakeNode(0)
    tm.node_available(0, n0, slots=1)
    await tm.submit(b"fn", (), {}, task_id="t1", target="head")
    assert n0.executed == ["t1"]


async def test_target_absent_rank_raises() -> None:
    tm = TaskManager()
    tm.node_available(0, FakeNode(0), slots=1)
    with pytest.raises(RuntimeError, match="no such node rank 7"):
        await tm.submit(b"fn", (), {}, task_id="t1", target=7)


async def test_target_busy_rank_queues_until_free() -> None:
    tm = TaskManager()
    node = FakeNode(1)
    node.gate = asyncio.Event()
    tm.node_available(1, node, slots=1)

    first = asyncio.ensure_future(tm.submit(b"fn", (), {}, task_id="t1", target=1))
    second = asyncio.ensure_future(tm.submit(b"fn", (), {}, task_id="t2", target=1))
    await asyncio.sleep(0.05)
    assert node.executed == ["t1"]
    node.gate.set()
    await asyncio.gather(first, second)
    assert node.executed == ["t1", "t2"]


async def test_broadcast_collects_per_node_results() -> None:
    tm = TaskManager()
    tm.node_available(0, FakeNode(0, result="a"), slots=1)
    tm.node_available(1, FakeNode(1, result="b"), slots=1)
    results = await tm.broadcast(b"fn", (), {}, task_id="b1")
    assert results == ["a", "b"]


async def test_broadcast_failure_becomes_value() -> None:
    tm = TaskManager()
    tm.node_available(0, FakeNode(0, result="a"), slots=1)
    tm.node_available(1, FakeNode(1, result=RuntimeError("bad")), slots=1)
    results = await tm.broadcast(b"fn", (), {}, task_id="b1")
    assert results[0] == "a"
    assert isinstance(results[1], RuntimeError)


async def test_broadcast_interrupted_node_becomes_runtime_error() -> None:
    tm = TaskManager()
    lost = FakeNode(1)
    lost.fail_times = 10
    tm.node_available(0, FakeNode(0, result="a"), slots=1)
    tm.node_available(1, lost, slots=1)
    results = await tm.broadcast(b"fn", (), {}, task_id="b1")
    assert results[0] == "a"
    assert isinstance(results[1], RuntimeError)
    assert "lost during broadcast" in str(results[1])


async def test_pressure_observer_receives_reports() -> None:
    reports: list[PressureReport] = []
    tm = TaskManager()
    tm.set_pressure_observer(reports.append)
    node = FakeNode(0)
    node.gate = asyncio.Event()
    tm.node_available(0, node, slots=1)

    task = asyncio.ensure_future(tm.submit(b"fn", (), {}, task_id="t1"))
    queued = asyncio.ensure_future(tm.submit(b"fn", (), {}, task_id="t2"))
    await asyncio.sleep(0.05)
    assert any(r.queued == 1 and r.inflight == 1 for r in reports)
    node.gate.set()
    await asyncio.gather(task, queued)
    assert reports[-1].queued == 0
    assert reports[-1].inflight == 0
    assert reports[-1].total_capacity == 1
    assert reports[-1].node_count == 1


async def test_node_unavailable_removes_from_rotation() -> None:
    tm = TaskManager()
    n0, n1 = FakeNode(0), FakeNode(1)
    tm.node_available(0, n0, slots=1)
    tm.node_available(1, n1, slots=1)
    tm.node_unavailable(0)
    await tm.submit(b"fn", (), {}, task_id="t1")
    await tm.submit(b"fn", (), {}, task_id="t2")
    assert n0.executed == []
    assert len(n1.executed) == 2
