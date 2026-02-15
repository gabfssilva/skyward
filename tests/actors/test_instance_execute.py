import asyncio

import pytest
from casty import ActorSystem, Behavior, Behaviors

from skyward.actors.messages import (
    Bootstrapped,
    Execute,
    Running,
    TaskResult,
)
from skyward.infra.worker import (
    ExecuteTask as WorkerExecuteTask,
    TaskSucceeded as WorkerTaskSucceeded,
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


class FakeClusterClient:
    def __init__(self, result: WorkerTaskSucceeded | None = None) -> None:
        self._result = result
        self.asks: list[bytes] = []

    async def ask(self, ref, msg_factory, *, timeout: float = 5.0):
        msg = msg_factory(ref)
        self.asks.append(msg.fn_bytes)
        return self._result or WorkerTaskSucceeded(result="ok", node_id=0)


@pytest.mark.asyncio
async def test_instance_pipe_to_self_executes_via_client(system):
    from skyward.actors.instance import instance_actor

    parent_msgs: list = []
    parent_ref = system.spawn(collector_behavior(parent_msgs), "parent")
    worker_ref = system.spawn(collector_behavior([]), "worker")
    fake_client = FakeClusterClient(WorkerTaskSucceeded(result=42, node_id=0))

    ref = system.spawn(
        instance_actor(
            instance_id="i-123",
            provider_ref=system.spawn(collector_behavior([]), "provider"),
            worker_ref=worker_ref,
            client=fake_client,
            parent=parent_ref,
            _skip_tunnel=True,
        ),
        "instance",
    )

    ref.tell(Running(ip="10.0.0.1"))
    await asyncio.sleep(0.1)
    ref.tell(Bootstrapped())
    await asyncio.sleep(0.1)

    ref.tell(Execute(fn_bytes=b"task-bytes", reply_to=ref))
    await asyncio.sleep(0.3)

    assert len(fake_client.asks) == 1
    assert fake_client.asks[0] == b"task-bytes"

    task_results = [m for m in parent_msgs if isinstance(m, TaskResult)]
    assert len(task_results) == 1
    assert task_results[0].value == 42


@pytest.mark.asyncio
async def test_instance_pipe_to_self_handles_failure(system):
    from skyward.actors.instance import instance_actor

    parent_msgs: list = []
    parent_ref = system.spawn(collector_behavior(parent_msgs), "parent")
    worker_ref = system.spawn(collector_behavior([]), "worker")

    class FailingClient:
        async def ask(self, ref, msg_factory, *, timeout: float = 5.0):
            raise ConnectionError("worker unreachable")

    ref = system.spawn(
        instance_actor(
            instance_id="i-123",
            provider_ref=system.spawn(collector_behavior([]), "provider"),
            worker_ref=worker_ref,
            client=FailingClient(),
            parent=parent_ref,
            _skip_tunnel=True,
        ),
        "instance",
    )

    ref.tell(Running(ip="10.0.0.1"))
    await asyncio.sleep(0.1)
    ref.tell(Bootstrapped())
    await asyncio.sleep(0.1)

    ref.tell(Execute(fn_bytes=b"task-bytes", reply_to=ref))
    await asyncio.sleep(0.3)

    task_results = [m for m in parent_msgs if isinstance(m, TaskResult)]
    assert len(task_results) == 1
    match task_results[0].value:
        case RuntimeError() as e:
            assert "worker unreachable" in str(e)
        case _:
            pytest.fail(f"Expected RuntimeError, got {task_results[0].value}")
