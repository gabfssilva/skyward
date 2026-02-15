import asyncio

import pytest
from casty import ActorSystem, Behavior, Behaviors

from skyward.actors.messages import (
    ExecuteOnNode,
    InstanceBecameReady,
    InstanceDied,
    InstanceLaunched,
    InstanceRunning,
    NodeBecameReady,
    NodeLost,
    Provision,
    SlotFreed,
    TaskResult,
)


def _launched(instance_id: str = "i-100", node_id: int = 0) -> InstanceLaunched:
    return InstanceLaunched(
        request_id="req-1", cluster_id="c1", node_id=node_id,
        provider="aws", instance_id=instance_id,
    )


def _running(ip: str = "10.0.0.1", instance_id: str = "i-100", node_id: int = 0) -> InstanceRunning:
    return InstanceRunning(
        request_id="req-1", cluster_id="c1", node_id=node_id,
        provider="aws", instance_id=instance_id,
        ip=ip, private_ip=None, ssh_port=22, spot=False,
    )


@pytest.fixture
def system():
    s = ActorSystem("test-node-v2")
    yield s
    asyncio.get_event_loop().run_until_complete(s.shutdown())


def collector_behavior(collected: list) -> Behavior:
    async def receive(ctx, msg):
        collected.append(msg)
        return Behaviors.same()
    return Behaviors.receive(receive)


@pytest.mark.asyncio
async def test_node_provision_sends_instance_requested(system):
    from skyward.actors.node import node_actor

    pool_msgs: list = []
    provider_msgs: list = []
    pool_ref = system.spawn(collector_behavior(pool_msgs), "pool")
    provider_ref = system.spawn(collector_behavior(provider_msgs), "provider")

    node_ref = system.spawn(node_actor(node_id=0, pool=pool_ref, task_manager=None), "node-0")
    node_ref.tell(Provision(cluster_id="c1", provider_ref=provider_ref))
    await asyncio.sleep(0.1)

    assert len(provider_msgs) == 1


@pytest.mark.asyncio
async def test_node_spawns_instance_on_launched(system):
    from skyward.actors.node import node_actor

    pool_msgs: list = []
    provider_msgs: list = []
    pool_ref = system.spawn(collector_behavior(pool_msgs), "pool")
    provider_ref = system.spawn(collector_behavior(provider_msgs), "provider")

    node_ref = system.spawn(node_actor(node_id=0, pool=pool_ref, task_manager=None), "node-0")
    node_ref.tell(Provision(cluster_id="c1", provider_ref=provider_ref))
    await asyncio.sleep(0.1)

    node_ref.tell(_launched())
    await asyncio.sleep(0.1)

    node_ref.tell(_running())
    await asyncio.sleep(0.1)

    node_ref.tell(InstanceBecameReady(instance_id="i-100", ip="10.0.0.1"))
    await asyncio.sleep(0.1)

    assert any(isinstance(m, NodeBecameReady) for m in pool_msgs)


@pytest.mark.asyncio
async def test_node_enqueues_tasks_during_provisioning(system):
    from skyward.actors.node import node_actor

    pool_msgs: list = []
    provider_msgs: list = []
    reply_msgs: list = []
    pool_ref = system.spawn(collector_behavior(pool_msgs), "pool")
    provider_ref = system.spawn(collector_behavior(provider_msgs), "provider")
    reply_ref = system.spawn(collector_behavior(reply_msgs), "reply")

    node_ref = system.spawn(node_actor(node_id=0, pool=pool_ref, task_manager=None), "node-0")
    node_ref.tell(Provision(cluster_id="c1", provider_ref=provider_ref))
    await asyncio.sleep(0.1)

    node_ref.tell(ExecuteOnNode(fn_bytes=b"task1", reply_to=reply_ref))
    await asyncio.sleep(0.1)

    assert len(reply_msgs) == 0


@pytest.mark.asyncio
async def test_node_preemption_recovery(system):
    from skyward.actors.node import node_actor

    pool_msgs: list = []
    provider_msgs: list = []
    tm_msgs: list = []
    pool_ref = system.spawn(collector_behavior(pool_msgs), "pool")
    provider_ref = system.spawn(collector_behavior(provider_msgs), "provider")
    tm_ref = system.spawn(collector_behavior(tm_msgs), "tm")

    node_ref = system.spawn(node_actor(node_id=0, pool=pool_ref, task_manager=tm_ref), "node-0")
    node_ref.tell(Provision(cluster_id="c1", provider_ref=provider_ref))
    await asyncio.sleep(0.1)

    node_ref.tell(_launched())
    node_ref.tell(_running())
    node_ref.tell(InstanceBecameReady(instance_id="i-100", ip="10.0.0.1"))
    await asyncio.sleep(0.2)

    pool_msgs.clear()
    provider_msgs.clear()

    node_ref.tell(InstanceDied(instance_id="i-100", reason="preempted"))
    await asyncio.sleep(0.1)

    assert any(isinstance(m, NodeLost) for m in pool_msgs)
    assert len(provider_msgs) >= 1


@pytest.mark.asyncio
async def test_node_intercepts_task_result(system):
    from skyward.actors.node import node_actor

    pool_msgs: list = []
    tm_msgs: list = []
    reply_msgs: list = []
    pool_ref = system.spawn(collector_behavior(pool_msgs), "pool")
    provider_ref = system.spawn(collector_behavior([]), "provider")
    tm_ref = system.spawn(collector_behavior(tm_msgs), "tm")
    reply_ref = system.spawn(collector_behavior(reply_msgs), "reply")

    node_ref = system.spawn(node_actor(node_id=0, pool=pool_ref, task_manager=tm_ref), "node-0")
    node_ref.tell(Provision(cluster_id="c1", provider_ref=provider_ref))
    await asyncio.sleep(0.1)

    node_ref.tell(_launched())
    node_ref.tell(_running())
    node_ref.tell(InstanceBecameReady(instance_id="i-100", ip="10.0.0.1"))
    await asyncio.sleep(0.2)

    node_ref.tell(ExecuteOnNode(fn_bytes=b"task1", reply_to=reply_ref))
    await asyncio.sleep(0.1)

    node_ref.tell(TaskResult(value="result-1", node_id=0))
    await asyncio.sleep(0.1)

    assert len(reply_msgs) == 1
    assert reply_msgs[0].value == "result-1"

    assert any(isinstance(m, SlotFreed) for m in tm_msgs)
