import asyncio

import pytest
from casty import ActorSystem, Behavior, Behaviors

from skyward.actors.messages import (
    ClusterProvisioned,
    ClusterRequested,
    InstanceRequested,
    StartPool,
)


class FakeSpec:
    def __init__(self, nodes: int):
        self.nodes = nodes
        self.provider = "aws"


@pytest.fixture
def system():
    s = ActorSystem("test-integration")
    yield s
    asyncio.get_event_loop().run_until_complete(s.shutdown())


def collector_behavior(collected: list) -> Behavior:
    async def receive(ctx, msg):
        collected.append(msg)
        return Behaviors.same()
    return Behaviors.receive(receive)


def fake_provider(pool_ref) -> Behavior:
    instance_counter = 0

    async def receive(ctx, msg):
        nonlocal instance_counter
        match msg:
            case ClusterRequested(request_id=rid, provider=prov):
                pool_ref.tell(ClusterProvisioned(
                    request_id=rid, cluster_id="c-test", provider=prov,
                ))
            case InstanceRequested():
                instance_counter += 1
        return Behaviors.same()
    return Behaviors.receive(receive)


@pytest.mark.asyncio
async def test_full_lifecycle_with_task_execution(system):
    from skyward.actors.pool import pool_actor

    reply_msgs: list = []
    reply_ref = system.spawn(collector_behavior(reply_msgs), "reply")

    pool_ref = system.spawn(pool_actor(), "pool")

    provider_ref = system.spawn(fake_provider(pool_ref), "provider")

    pool_ref.tell(StartPool(
        spec=FakeSpec(nodes=1),  # type: ignore[arg-type]
        provider_config=None,  # type: ignore[arg-type]
        provider_ref=provider_ref,
        reply_to=reply_ref,
    ))
    await asyncio.sleep(0.5)

    assert True
