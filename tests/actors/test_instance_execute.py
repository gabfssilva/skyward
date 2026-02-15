import asyncio

import pytest
from casty import ActorSystem, Behavior, Behaviors

from skyward.actors.messages import (
    Bootstrapped,
    Execute,
    Running,
    TaskResult,
)


@pytest.fixture
def system():
    s = ActorSystem("test-instance-exec")
    yield s
    asyncio.get_event_loop().run_until_complete(s.shutdown())


def collector_behavior(collected: list) -> Behavior:
    async def receive(ctx, msg):
        collected.append(msg)
        return Behaviors.same()
    return Behaviors.receive(receive)


@pytest.mark.asyncio
async def test_instance_forwards_execute_to_worker(system):
    from skyward.actors.instance import instance_actor

    parent_msgs: list = []
    worker_msgs: list = []
    parent_ref = system.spawn(collector_behavior(parent_msgs), "parent")
    worker_ref = system.spawn(collector_behavior(worker_msgs), "worker")

    class FakeClusterClient:
        def lookup(self, key):
            return worker_ref

    ref = system.spawn(
        instance_actor(
            instance_id="i-123",
            provider_ref=system.spawn(collector_behavior([]), "provider"),
            cluster_client=FakeClusterClient(),
            parent=parent_ref,
            _skip_tunnel=True,
        ),
        "instance",
    )

    ref.tell(Running(ip="10.0.0.1"))
    await asyncio.sleep(0.1)
    ref.tell(Bootstrapped())
    await asyncio.sleep(0.1)

    node_ref = system.spawn(collector_behavior([]), "node")
    ref.tell(Execute(fn_bytes=b"task-bytes", reply_to=node_ref))
    await asyncio.sleep(0.1)

    assert len(worker_msgs) == 1
