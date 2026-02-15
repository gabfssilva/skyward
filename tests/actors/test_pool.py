from __future__ import annotations

import asyncio

import pytest
from casty import ActorContext, ActorRef, ActorSystem, Behavior, Behaviors

from skyward.actors.messages import (
    ClusterConnected,
    ClusterProvisioned,
    ClusterRequested,
    InstanceMetadata,
    NodeBecameReady,
    NodeLost,
    PoolMsg,
    PoolStarted,
    PoolStopped,
    ShutdownCompleted,
    ShutdownRequested,
    StartPool,
    StopPool,
    SubmitTask,
)
from skyward.actors.pool import pool_actor
from skyward.api.spec import Image, PoolSpec

pytestmark = pytest.mark.xdist_group("pool-actor")


def probe_actor(results: list) -> Behavior:
    async def receive(ctx, msg):
        results.append(msg)
        match msg:
            case ShutdownRequested(reply_to=reply_to) if reply_to is not None:
                reply_to.tell(ShutdownCompleted(cluster_id=msg.cluster_id))
        return Behaviors.same()
    return Behaviors.receive(receive)


def reply_probe[T](results: list[T]) -> Behavior[T]:
    async def receive(ctx: ActorContext[T], msg: T) -> Behavior[T]:
        results.append(msg)
        return Behaviors.same()

    return Behaviors.receive(receive)


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def actor_system(event_loop):
    async def _start():
        system = ActorSystem("test-pool")
        await system.__aenter__()
        return system

    system = event_loop.run_until_complete(_start())
    yield system, event_loop
    event_loop.run_until_complete(system.__aexit__(None, None, None))


def _make_spec(nodes: int = 2) -> PoolSpec:
    return PoolSpec(
        nodes=nodes,
        accelerator="A100",
        region="us-east-1",
        image=Image(),
    )


def _make_instance(node: int, instance_id: str = "") -> InstanceMetadata:
    iid = instance_id or f"i-{node:04d}"
    return InstanceMetadata(
        id=iid,
        node=node,
        provider="aws",
        ip=f"10.0.0.{node + 1}",
    )


def test_pool_emits_cluster_requested_on_start(actor_system):
    system, loop = actor_system
    provider_events: list = []
    reply_msgs: list[PoolStarted] = []

    async def _test():
        provider_ref = system.spawn(probe_actor(provider_events), "provider-1")
        reply_ref = system.spawn(reply_probe(reply_msgs), "reply-probe-1")

        pool_ref = system.spawn(
            pool_actor(),
            "pool-1",
        )

        spec = _make_spec(nodes=2)
        pool_ref.tell(StartPool(
            spec=spec,
            provider_config=None,  # type: ignore[arg-type]
            provider_ref=provider_ref,
            reply_to=reply_ref,
        ))

        await asyncio.sleep(0.5)

        cluster_requests = [e for e in provider_events if isinstance(e, ClusterRequested)]
        assert len(cluster_requests) == 1
        assert cluster_requests[0].spec.nodes == 2

    loop.run_until_complete(_test())


def test_pool_full_lifecycle(actor_system):
    system, loop = actor_system
    provider_events: list = []
    reply_msgs: list = []

    async def _test():
        provider_ref = system.spawn(probe_actor(provider_events), "provider-2")
        reply_ref = system.spawn(reply_probe(reply_msgs), "reply-probe-2")

        pool_ref: ActorRef[PoolMsg] = system.spawn(
            pool_actor(),
            "pool-2",
        )

        spec = _make_spec(nodes=2)
        pool_ref.tell(StartPool(
            spec=spec,
            provider_config=None,  # type: ignore[arg-type]
            provider_ref=provider_ref,
            reply_to=reply_ref,
        ))

        await asyncio.sleep(0.3)

        cluster_requests = [e for e in provider_events if isinstance(e, ClusterRequested)]
        assert len(cluster_requests) == 1
        request_id = cluster_requests[0].request_id

        pool_ref.tell(ClusterProvisioned(
            request_id=request_id,
            cluster_id="c-lifecycle",
            provider="aws",
        ))

        await asyncio.sleep(0.3)

        info_0 = _make_instance(0)
        info_1 = _make_instance(1)

        pool_ref.tell(NodeBecameReady(node_id=0, instance=info_0))
        pool_ref.tell(NodeBecameReady(node_id=1, instance=info_1))
        await asyncio.sleep(0.3)

        started_replies = [r for r in reply_msgs if isinstance(r, PoolStarted)]
        assert len(started_replies) == 1
        assert started_replies[0].cluster_id == "c-lifecycle"
        assert len(started_replies[0].instances) == 2

    loop.run_until_complete(_test())


def test_pool_submit_task_delegates_to_task_manager(actor_system):
    system, loop = actor_system
    provider_events: list = []
    start_replies: list = []
    task_replies: list = []

    async def _test():
        provider_ref = system.spawn(probe_actor(provider_events), "provider-3")
        start_reply_ref = system.spawn(reply_probe(start_replies), "start-reply-3")
        task_reply_ref = system.spawn(reply_probe(task_replies), "task-reply-3")

        pool_ref: ActorRef[PoolMsg] = system.spawn(
            pool_actor(),
            "pool-3",
        )

        spec = _make_spec(nodes=1)
        pool_ref.tell(StartPool(
            spec=spec,
            provider_config=None,  # type: ignore[arg-type]
            provider_ref=provider_ref,
            reply_to=start_reply_ref,
        ))
        await asyncio.sleep(0.3)

        request_id = [e for e in provider_events if isinstance(e, ClusterRequested)][0].request_id
        pool_ref.tell(ClusterProvisioned(
            request_id=request_id,
            cluster_id="c-submit",
            provider="aws",
        ))
        await asyncio.sleep(0.3)

        pool_ref.tell(NodeBecameReady(node_id=0, instance=_make_instance(0)))
        await asyncio.sleep(0.3)

        pool_ref.tell(SubmitTask(fn_bytes=b"task-data", reply_to=task_reply_ref))
        await asyncio.sleep(0.3)

    loop.run_until_complete(_test())


def test_pool_cluster_connected_distributes_worker_refs(actor_system):
    system, loop = actor_system
    provider_events: list = []
    start_replies: list = []
    node_msgs: list = []

    async def _test():
        provider_ref = system.spawn(probe_actor(provider_events), "provider-8")
        start_reply_ref = system.spawn(reply_probe(start_replies), "start-reply-8")

        pool_ref: ActorRef[PoolMsg] = system.spawn(
            pool_actor(),
            "pool-8",
        )

        spec = _make_spec(nodes=2)
        pool_ref.tell(StartPool(
            spec=spec,
            provider_config=None,  # type: ignore[arg-type]
            provider_ref=provider_ref,
            reply_to=start_reply_ref,
        ))
        await asyncio.sleep(0.3)

        request_id = [e for e in provider_events if isinstance(e, ClusterRequested)][0].request_id
        pool_ref.tell(ClusterProvisioned(
            request_id=request_id,
            cluster_id="c-cc",
            provider="aws",
        ))
        await asyncio.sleep(0.3)

        pool_ref.tell(NodeBecameReady(node_id=0, instance=_make_instance(0)))
        pool_ref.tell(NodeBecameReady(node_id=1, instance=_make_instance(1)))
        await asyncio.sleep(0.3)

        worker_0 = system.spawn(reply_probe(node_msgs), "fake-worker-0")
        worker_1 = system.spawn(reply_probe(node_msgs), "fake-worker-1")
        pool_ref.tell(ClusterConnected(worker_refs=((0, worker_0), (1, worker_1)), client=None))  # type: ignore[arg-type]
        await asyncio.sleep(0.3)

    loop.run_until_complete(_test())


def test_pool_stop_emits_shutdown(actor_system):
    system, loop = actor_system
    provider_events: list = []
    start_replies: list = []
    stop_replies: list[PoolStopped] = []

    async def _test():
        provider_ref = system.spawn(probe_actor(provider_events), "provider-5")
        start_reply_ref = system.spawn(reply_probe(start_replies), "start-reply-5")
        stop_reply_ref = system.spawn(reply_probe(stop_replies), "stop-reply-5")

        pool_ref: ActorRef[PoolMsg] = system.spawn(
            pool_actor(),
            "pool-5",
        )

        spec = _make_spec(nodes=1)
        pool_ref.tell(StartPool(
            spec=spec,
            provider_config=None,  # type: ignore[arg-type]
            provider_ref=provider_ref,
            reply_to=start_reply_ref,
        ))
        await asyncio.sleep(0.3)

        request_id = [e for e in provider_events if isinstance(e, ClusterRequested)][0].request_id
        pool_ref.tell(ClusterProvisioned(
            request_id=request_id,
            cluster_id="c-stop",
            provider="aws",
        ))
        await asyncio.sleep(0.3)

        pool_ref.tell(NodeBecameReady(node_id=0, instance=_make_instance(0)))
        await asyncio.sleep(0.3)

        pool_ref.tell(StopPool(reply_to=stop_reply_ref))
        await asyncio.sleep(0.3)

        shutdown_events = [e for e in provider_events if isinstance(e, ShutdownRequested)]
        assert len(shutdown_events) == 1
        assert shutdown_events[0].cluster_id == "c-stop"

        assert len(stop_replies) == 1
        assert isinstance(stop_replies[0], PoolStopped)

    loop.run_until_complete(_test())


def test_pool_node_lost_in_ready_state(actor_system):
    system, loop = actor_system
    provider_events: list = []
    start_replies: list = []

    async def _test():
        provider_ref = system.spawn(probe_actor(provider_events), "provider-6")
        start_reply_ref = system.spawn(reply_probe(start_replies), "start-reply-6")

        pool_ref: ActorRef[PoolMsg] = system.spawn(
            pool_actor(),
            "pool-6",
        )

        spec = _make_spec(nodes=2)
        pool_ref.tell(StartPool(
            spec=spec,
            provider_config=None,  # type: ignore[arg-type]
            provider_ref=provider_ref,
            reply_to=start_reply_ref,
        ))
        await asyncio.sleep(0.3)

        request_id = [e for e in provider_events if isinstance(e, ClusterRequested)][0].request_id
        pool_ref.tell(ClusterProvisioned(
            request_id=request_id,
            cluster_id="c-preempt",
            provider="aws",
        ))
        await asyncio.sleep(0.3)

        pool_ref.tell(NodeBecameReady(node_id=0, instance=_make_instance(0)))
        pool_ref.tell(NodeBecameReady(node_id=1, instance=_make_instance(1)))
        await asyncio.sleep(0.5)

        pool_ref.tell(NodeLost(node_id=1, reason="spot-interruption"))
        await asyncio.sleep(0.3)

        new_info_1 = InstanceMetadata(
            id="i-new-1",
            node=1,
            provider="aws",
            ip="10.0.0.10",
        )
        pool_ref.tell(NodeBecameReady(node_id=1, instance=new_info_1))
        await asyncio.sleep(0.3)

    loop.run_until_complete(_test())
