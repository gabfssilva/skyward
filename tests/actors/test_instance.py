import asyncio

import pytest
from casty import ActorSystem, Behavior, Behaviors

from skyward.actors.messages import (
    Bootstrapped,
    Bootstrapping,
    Execute,
    InstanceBecameReady,
    InstanceDied,
    Log,
    Metric,
    Preempted,
    Running,
)


@pytest.fixture
def system():
    s = ActorSystem("test-instance")
    yield s
    asyncio.get_event_loop().run_until_complete(s.shutdown())


def collector_behavior(collected: list) -> Behavior:
    async def receive(ctx, msg):
        collected.append(msg)
        return Behaviors.same()
    return Behaviors.receive(receive)


@pytest.mark.asyncio
async def test_instance_waiting_receives_running(system):
    from skyward.actors.instance import instance_actor

    parent_msgs: list = []
    parent_ref = system.spawn(collector_behavior(parent_msgs), "parent")

    ref = system.spawn(
        instance_actor(
            instance_id="i-123",
            provider_ref=system.spawn(collector_behavior([]), "provider"),
            worker_ref=None,
            parent=parent_ref,
            _skip_tunnel=True,
        ),
        "instance",
    )

    ref.tell(Running(ip="10.0.0.1"))
    await asyncio.sleep(0.1)

    # Instance should transition to starting/bootstrapping
    # No parent notification yet (not ready)
    assert len(parent_msgs) == 0


@pytest.mark.asyncio
async def test_instance_preempted_in_waiting(system):
    from skyward.actors.instance import instance_actor

    parent_msgs: list = []
    parent_ref = system.spawn(collector_behavior(parent_msgs), "parent")

    ref = system.spawn(
        instance_actor(
            instance_id="i-123",
            provider_ref=system.spawn(collector_behavior([]), "provider"),
            worker_ref=None,
            parent=parent_ref,
            _skip_tunnel=True,
        ),
        "instance",
    )

    ref.tell(Preempted())
    await asyncio.sleep(0.1)

    assert len(parent_msgs) == 1
    assert isinstance(parent_msgs[0], InstanceDied)
    assert parent_msgs[0].instance_id == "i-123"


@pytest.mark.asyncio
async def test_instance_full_lifecycle(system):
    from skyward.actors.instance import instance_actor

    parent_msgs: list = []
    provider_msgs: list = []
    parent_ref = system.spawn(collector_behavior(parent_msgs), "parent")
    provider_ref = system.spawn(collector_behavior(provider_msgs), "provider")

    ref = system.spawn(
        instance_actor(
            instance_id="i-123",
            provider_ref=provider_ref,
            worker_ref=None,
            parent=parent_ref,
            _skip_tunnel=True,
        ),
        "instance",
    )

    ref.tell(Running(ip="10.0.0.1"))
    await asyncio.sleep(0.1)

    ref.tell(Bootstrapping(phase="setup", status="running"))
    await asyncio.sleep(0.1)

    ref.tell(Bootstrapped())
    await asyncio.sleep(0.1)

    assert any(isinstance(m, InstanceBecameReady) for m in parent_msgs)
    ready_msg = next(m for m in parent_msgs if isinstance(m, InstanceBecameReady))
    assert ready_msg.instance_id == "i-123"
    assert ready_msg.ip == "10.0.0.1"


@pytest.mark.asyncio
async def test_instance_preempted_in_ready(system):
    from skyward.actors.instance import instance_actor

    parent_msgs: list = []
    parent_ref = system.spawn(collector_behavior(parent_msgs), "parent")

    ref = system.spawn(
        instance_actor(
            instance_id="i-123",
            provider_ref=system.spawn(collector_behavior([]), "provider"),
            worker_ref=None,
            parent=parent_ref,
            _skip_tunnel=True,
        ),
        "instance",
    )

    ref.tell(Running(ip="10.0.0.1"))
    await asyncio.sleep(0.1)
    ref.tell(Bootstrapped())
    await asyncio.sleep(0.1)
    parent_msgs.clear()

    ref.tell(Preempted())
    await asyncio.sleep(0.1)

    assert len(parent_msgs) == 1
    assert isinstance(parent_msgs[0], InstanceDied)
