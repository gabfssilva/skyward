import asyncio
from unittest.mock import AsyncMock

import pytest
from casty import ActorSystem, Behavior, Behaviors

from skyward.actors.messages import (
    Bootstrapped,
    Bootstrapping,
    InstanceBecameReady,
    Running,
)


@pytest.fixture
def system():
    s = ActorSystem("test-instance-stream")
    yield s
    asyncio.get_event_loop().run_until_complete(s.shutdown())


def collector_behavior(collected: list) -> Behavior:
    async def receive(ctx, msg):
        collected.append(msg)
        return Behaviors.same()
    return Behaviors.receive(receive)


@pytest.mark.asyncio
async def test_instance_streams_events_after_running(system):
    from skyward.actors.instance import instance_actor

    parent_msgs: list = []
    provider_msgs: list = []
    parent_ref = system.spawn(collector_behavior(parent_msgs), "parent")
    provider_ref = system.spawn(collector_behavior(provider_msgs), "provider")

    mock_tunnel_factory = AsyncMock()

    ref = system.spawn(
        instance_actor(
            instance_id="i-123",
            provider_ref=provider_ref,
            worker_ref=None,
            parent=parent_ref,
            _tunnel_factory=mock_tunnel_factory,
        ),
        "instance",
    )

    ref.tell(Running(ip="10.0.0.1"))
    await asyncio.sleep(0.2)

    # Simulate events arriving from stream
    ref.tell(Bootstrapping(phase="deps", status="running"))
    ref.tell(Bootstrapping(phase="deps", status="completed", elapsed=5.0))
    ref.tell(Bootstrapped())
    await asyncio.sleep(0.2)

    assert any(isinstance(m, InstanceBecameReady) for m in parent_msgs)
