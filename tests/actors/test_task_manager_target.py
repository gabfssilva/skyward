"""Node-targeted dispatch in the task manager (SubmitTask.target)."""

from __future__ import annotations

import asyncio

import pytest
from casty import ActorContext, ActorSystem, Behavior, Behaviors

from skyward.actors.messages import (
    ExecuteOnNode,
    NodeAvailable,
    SubmitTask,
    TaskFailed,
)
from skyward.actors.task_manager.actor import task_manager_actor

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


def _probe(log: list[int], node_id: int) -> Behavior:
    async def receive(ctx: ActorContext, msg: object) -> Behavior:
        if isinstance(msg, ExecuteOnNode):
            log.append(node_id)
        return Behaviors.same()
    return Behaviors.receive(receive)


def _caller(log: list[object]) -> Behavior:
    async def receive(ctx: ActorContext, msg: object) -> Behavior:
        log.append(msg)
        return Behaviors.same()
    return Behaviors.receive(receive)


async def _two_node_tm(system: ActorSystem, dispatched: list[int]):
    tm = system.spawn(task_manager_actor(), "tm")
    n0 = system.spawn(_probe(dispatched, 0), "n0")
    n1 = system.spawn(_probe(dispatched, 1), "n1")
    tm.tell(NodeAvailable(node_id=0, node_ref=n0, slots=1))
    tm.tell(NodeAvailable(node_id=1, node_ref=n1, slots=1))
    await asyncio.sleep(0.05)
    return tm


async def test_target_int_routes_to_rank() -> None:
    async with ActorSystem("tm-target-int") as system:
        dispatched: list[int] = []
        tm = await _two_node_tm(system, dispatched)
        caller = system.spawn(_caller([]), "caller")
        tm.tell(SubmitTask(fn=b"x", args=(), kwargs={}, reply_to=caller, task_id="t", target=1))
        await asyncio.sleep(0.05)
        assert dispatched == [1]


async def test_target_head_routes_to_rank_zero() -> None:
    async with ActorSystem("tm-target-head") as system:
        dispatched: list[int] = []
        tm = await _two_node_tm(system, dispatched)
        caller = system.spawn(_caller([]), "caller")
        tm.tell(SubmitTask(fn=b"x", args=(), kwargs={}, reply_to=caller, task_id="t", target="head"))
        await asyncio.sleep(0.05)
        assert dispatched == [0]


async def test_target_absent_rank_fails_fast() -> None:
    async with ActorSystem("tm-target-absent") as system:
        dispatched: list[int] = []
        tm = await _two_node_tm(system, dispatched)
        replies: list[object] = []
        caller = system.spawn(_caller(replies), "caller")
        tm.tell(SubmitTask(fn=b"x", args=(), kwargs={}, reply_to=caller, task_id="t", target=99))
        await asyncio.sleep(0.05)
        assert dispatched == []
        assert len(replies) == 1
        assert isinstance(replies[0], TaskFailed)
        assert replies[0].node_id == 99


async def test_target_none_round_robins() -> None:
    async with ActorSystem("tm-target-none") as system:
        dispatched: list[int] = []
        tm = await _two_node_tm(system, dispatched)
        caller = system.spawn(_caller([]), "caller")
        tm.tell(SubmitTask(fn=b"x", args=(), kwargs={}, reply_to=caller, task_id="a"))
        tm.tell(SubmitTask(fn=b"x", args=(), kwargs={}, reply_to=caller, task_id="b"))
        await asyncio.sleep(0.05)
        assert sorted(dispatched) == [0, 1]


# ── unit: _pick_target / _drain_queue ────────────────────────────


def test_pick_target_present_and_absent() -> None:
    from types import MappingProxyType
    from unittest.mock import MagicMock

    from skyward.actors.messages import NodeSlots
    from skyward.actors.task_manager.state import _pick_target

    nodes = MappingProxyType({0: NodeSlots(MagicMock(), 1, 0)})
    assert _pick_target(nodes, "head") == 0
    assert _pick_target(nodes, 0) == 0
    assert _pick_target(nodes, 5) is None


def test_drain_queue_targeted_busy_stays_queued() -> None:
    from types import MappingProxyType
    from unittest.mock import MagicMock

    from skyward.actors.messages import NodeSlots
    from skyward.actors.task_manager.state import _drain_queue

    nodes = MappingProxyType({1: NodeSlots(MagicMock(), 1, 1)})
    task = SubmitTask(fn=b"x", args=(), kwargs={}, reply_to=MagicMock(), task_id="t", target=1)
    remaining, _nodes, _rr, _inf, _nt = _drain_queue(
        (task,), nodes, 0, MagicMock(), MappingProxyType({}), MappingProxyType({}),
    )
    assert remaining == (task,)


def test_drain_queue_targeted_free_dispatches() -> None:
    from types import MappingProxyType
    from unittest.mock import MagicMock

    from skyward.actors.messages import NodeSlots
    from skyward.actors.task_manager.state import _drain_queue

    nodes = MappingProxyType({1: NodeSlots(MagicMock(), 1, 0)})
    task = SubmitTask(fn=b"x", args=(), kwargs={}, reply_to=MagicMock(), task_id="t", target=1)
    remaining, new_nodes, _rr, _inf, _nt = _drain_queue(
        (task,), nodes, 0, MagicMock(), MappingProxyType({}), MappingProxyType({}),
    )
    assert remaining == ()
    assert new_nodes[1].used == 1
