import asyncio

import pytest
from casty import ActorSystem, Behavior, Behaviors

from skyward.actors.messages import (
    ExecuteOnNode,
    NodeAvailable,
    NodeUnavailable,
    SlotFreed,
    SubmitBroadcast,
    SubmitTask,
)


@pytest.fixture
def system():
    s = ActorSystem("test-task-manager")
    yield s
    asyncio.get_event_loop().run_until_complete(s.shutdown())


def collector_behavior(collected: list) -> Behavior:
    async def receive(ctx, msg):
        collected.append(msg)
        return Behaviors.same()
    return Behaviors.receive(receive)


@pytest.mark.asyncio
async def test_submit_task_routes_to_available_node(system):
    from skyward.actors.task_manager import task_manager_actor

    node_msgs: list = []
    reply_msgs: list = []

    node_ref = system.spawn(collector_behavior(node_msgs), "node-0")
    reply_ref = system.spawn(collector_behavior(reply_msgs), "reply")
    tm_ref = system.spawn(task_manager_actor(), "tm")

    tm_ref.tell(NodeAvailable(node_id=0, node_ref=node_ref, slots=2))
    await asyncio.sleep(0.1)

    tm_ref.tell(SubmitTask(fn_bytes=b"task1", reply_to=reply_ref))
    await asyncio.sleep(0.1)

    assert len(node_msgs) == 1
    assert isinstance(node_msgs[0], ExecuteOnNode)
    assert node_msgs[0].fn_bytes == b"task1"


@pytest.mark.asyncio
async def test_submit_task_queues_when_no_slots(system):
    from skyward.actors.task_manager import task_manager_actor

    node_msgs: list = []
    reply_ref = system.spawn(collector_behavior([]), "reply")
    node_ref = system.spawn(collector_behavior(node_msgs), "node-0")
    tm_ref = system.spawn(task_manager_actor(), "tm")

    tm_ref.tell(NodeAvailable(node_id=0, node_ref=node_ref, slots=1))
    await asyncio.sleep(0.1)

    tm_ref.tell(SubmitTask(fn_bytes=b"task1", reply_to=reply_ref))
    tm_ref.tell(SubmitTask(fn_bytes=b"task2", reply_to=reply_ref))
    await asyncio.sleep(0.1)

    assert len(node_msgs) == 1  # only 1 slot

    tm_ref.tell(SlotFreed(node_id=0))
    await asyncio.sleep(0.1)

    assert len(node_msgs) == 2  # queue drained


@pytest.mark.asyncio
async def test_submit_task_queues_when_no_nodes(system):
    from skyward.actors.task_manager import task_manager_actor

    node_msgs: list = []
    reply_ref = system.spawn(collector_behavior([]), "reply")
    node_ref = system.spawn(collector_behavior(node_msgs), "node-0")
    tm_ref = system.spawn(task_manager_actor(), "tm")

    tm_ref.tell(SubmitTask(fn_bytes=b"task1", reply_to=reply_ref))
    await asyncio.sleep(0.1)
    assert len(node_msgs) == 0

    tm_ref.tell(NodeAvailable(node_id=0, node_ref=node_ref, slots=2))
    await asyncio.sleep(0.1)
    assert len(node_msgs) == 1  # drained on node join


@pytest.mark.asyncio
async def test_broadcast_sends_to_all_nodes(system):
    from skyward.actors.task_manager import task_manager_actor

    node0_msgs: list = []
    node1_msgs: list = []
    reply_ref = system.spawn(collector_behavior([]), "reply")
    node0_ref = system.spawn(collector_behavior(node0_msgs), "node-0")
    node1_ref = system.spawn(collector_behavior(node1_msgs), "node-1")
    tm_ref = system.spawn(task_manager_actor(), "tm")

    tm_ref.tell(NodeAvailable(node_id=0, node_ref=node0_ref, slots=1))
    tm_ref.tell(NodeAvailable(node_id=1, node_ref=node1_ref, slots=1))
    await asyncio.sleep(0.1)

    tm_ref.tell(SubmitBroadcast(fn_bytes=b"bcast", reply_to=reply_ref))
    await asyncio.sleep(0.1)

    assert len(node0_msgs) == 1
    assert len(node1_msgs) == 1


@pytest.mark.asyncio
async def test_node_unavailable_removes_node(system):
    from skyward.actors.task_manager import task_manager_actor

    node_msgs: list = []
    reply_ref = system.spawn(collector_behavior([]), "reply")
    node_ref = system.spawn(collector_behavior(node_msgs), "node-0")
    tm_ref = system.spawn(task_manager_actor(), "tm")

    tm_ref.tell(NodeAvailable(node_id=0, node_ref=node_ref, slots=2))
    tm_ref.tell(NodeUnavailable(node_id=0))
    await asyncio.sleep(0.1)

    tm_ref.tell(SubmitTask(fn_bytes=b"task1", reply_to=reply_ref))
    await asyncio.sleep(0.1)

    assert len(node_msgs) == 0  # queued, no available nodes


@pytest.mark.asyncio
async def test_round_robin_across_nodes(system):
    from skyward.actors.task_manager import task_manager_actor

    node0_msgs: list = []
    node1_msgs: list = []
    reply_ref = system.spawn(collector_behavior([]), "reply")
    node0_ref = system.spawn(collector_behavior(node0_msgs), "node-0")
    node1_ref = system.spawn(collector_behavior(node1_msgs), "node-1")
    tm_ref = system.spawn(task_manager_actor(), "tm")

    tm_ref.tell(NodeAvailable(node_id=0, node_ref=node0_ref, slots=2))
    tm_ref.tell(NodeAvailable(node_id=1, node_ref=node1_ref, slots=2))
    await asyncio.sleep(0.1)

    tm_ref.tell(SubmitTask(fn_bytes=b"t1", reply_to=reply_ref))
    tm_ref.tell(SubmitTask(fn_bytes=b"t2", reply_to=reply_ref))
    await asyncio.sleep(0.1)

    assert len(node0_msgs) == 1
    assert len(node1_msgs) == 1
