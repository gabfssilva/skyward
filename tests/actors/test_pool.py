from __future__ import annotations

import asyncio

import pytest
from casty import ActorContext, ActorRef, ActorSystem, Behavior, Behaviors

from skyward.actors.messages import (
    BroadcastResult,
    BroadcastTask,
    ClusterProvisioned,
    ClusterRequested,
    ExecuteResult,
    ExecuteTask,
    InstanceBootstrapped,
    InstanceMetadata,
    InstancePreempted,
    InstanceProvisioned,
    NodeBecameReady,
    PoolMsg,
    PoolStarted,
    PoolStopped,
    ShutdownRequested,
    StartPool,
    StopPool,
)
from skyward.actors.pool import pool_actor
from skyward.api.spec import Image, PoolSpec

pytestmark = pytest.mark.xdist_group("pool-actor")


# =============================================================================
# Probe actors for capturing messages
# =============================================================================


def probe_actor(results: list) -> Behavior:
    async def receive(ctx, msg):
        results.append(msg)
        return Behaviors.same()
    return Behaviors.receive(receive)


def reply_probe[T](results: list[T]) -> Behavior[T]:
    async def receive(ctx: ActorContext[T], msg: T) -> Behavior[T]:
        results.append(msg)
        return Behaviors.same()

    return Behaviors.receive(receive)


# =============================================================================
# Fixtures
# =============================================================================


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


# =============================================================================
# Tests: message construction
# =============================================================================


def test_pool_messages_importable():
    stopped = PoolStopped()
    assert stopped is not None

    result = ExecuteResult(value=42, node_id=0)
    assert result.value == 42
    assert result.node_id == 0

    broadcast = BroadcastResult(values=(1, 2, 3))
    assert len(broadcast.values) == 3


# =============================================================================
# Tests: idle -> requesting transition
# =============================================================================


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
            provider_config=None,
            provider_ref=provider_ref,
            reply_to=reply_ref,
        ))

        await asyncio.sleep(0.5)

        cluster_requests = [e for e in provider_events if isinstance(e, ClusterRequested)]
        assert len(cluster_requests) == 1
        assert cluster_requests[0].provider == "aws"
        assert cluster_requests[0].spec.nodes == 2

    loop.run_until_complete(_test())


# =============================================================================
# Tests: requesting -> provisioning -> ready
# =============================================================================


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
            provider_config=None,
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

        await asyncio.sleep(0.5)

        info_0 = _make_instance(0)
        info_1 = _make_instance(1)

        pool_ref.tell(InstanceProvisioned(request_id="r-0", instance=info_0))
        pool_ref.tell(InstanceProvisioned(request_id="r-1", instance=info_1))
        await asyncio.sleep(0.3)

        pool_ref.tell(InstanceBootstrapped(instance=info_0))
        pool_ref.tell(InstanceBootstrapped(instance=info_1))
        await asyncio.sleep(0.3)

        pool_ref.tell(NodeBecameReady(node_id=0, instance=info_0))
        pool_ref.tell(NodeBecameReady(node_id=1, instance=info_1))
        await asyncio.sleep(0.5)

        started_replies = [r for r in reply_msgs if isinstance(r, PoolStarted)]
        assert len(started_replies) == 1
        assert started_replies[0].cluster_id == "c-lifecycle"
        assert len(started_replies[0].instances) == 2

    loop.run_until_complete(_test())


# =============================================================================
# Tests: ready state - execute and broadcast
# =============================================================================


def test_pool_execute_task_in_ready_state(actor_system):
    system, loop = actor_system
    provider_events: list = []
    start_replies: list = []
    exec_replies: list[ExecuteResult] = []

    async def _test():
        provider_ref = system.spawn(probe_actor(provider_events), "provider-3")
        start_reply_ref = system.spawn(reply_probe(start_replies), "start-reply-3")
        exec_reply_ref = system.spawn(reply_probe(exec_replies), "exec-reply-3")

        pool_ref: ActorRef[PoolMsg] = system.spawn(
            pool_actor(),
            "pool-3",
        )

        spec = _make_spec(nodes=1)
        pool_ref.tell(StartPool(
            spec=spec,
            provider_config=None,
            provider_ref=provider_ref,
            reply_to=start_reply_ref,
        ))
        await asyncio.sleep(0.3)

        request_id = [e for e in provider_events if isinstance(e, ClusterRequested)][0].request_id
        pool_ref.tell(ClusterProvisioned(
            request_id=request_id,
            cluster_id="c-exec",
            provider="aws",
        ))
        await asyncio.sleep(0.3)

        info_0 = _make_instance(0)
        pool_ref.tell(NodeBecameReady(node_id=0, instance=info_0))
        await asyncio.sleep(0.3)

        pool_ref.tell(ExecuteTask(
            fn=lambda x, y: x + y,
            args=(3, 4),
            kwargs={},
            node=0,
            reply_to=exec_reply_ref,
        ))
        await asyncio.sleep(0.3)

        assert len(exec_replies) == 1
        assert exec_replies[0].value == 7
        assert exec_replies[0].node_id == 0

    loop.run_until_complete(_test())


def test_pool_broadcast_task(actor_system):
    system, loop = actor_system
    provider_events: list = []
    start_replies: list = []
    broadcast_replies: list[BroadcastResult] = []

    async def _test():
        provider_ref = system.spawn(probe_actor(provider_events), "provider-4")
        start_reply_ref = system.spawn(reply_probe(start_replies), "start-reply-4")
        bcast_reply_ref = system.spawn(reply_probe(broadcast_replies), "bcast-reply-4")

        pool_ref: ActorRef[PoolMsg] = system.spawn(
            pool_actor(),
            "pool-4",
        )

        spec = _make_spec(nodes=3)
        pool_ref.tell(StartPool(
            spec=spec,
            provider_config=None,
            provider_ref=provider_ref,
            reply_to=start_reply_ref,
        ))
        await asyncio.sleep(0.3)

        request_id = [e for e in provider_events if isinstance(e, ClusterRequested)][0].request_id
        pool_ref.tell(ClusterProvisioned(
            request_id=request_id,
            cluster_id="c-bcast",
            provider="aws",
        ))
        await asyncio.sleep(0.3)

        for i in range(3):
            pool_ref.tell(NodeBecameReady(node_id=i, instance=_make_instance(i)))
        await asyncio.sleep(0.5)

        counter = {"n": 0}

        def counting_fn():
            counter["n"] += 1
            return counter["n"]

        pool_ref.tell(BroadcastTask(
            fn=counting_fn,
            args=(),
            kwargs={},
            reply_to=bcast_reply_ref,
        ))
        await asyncio.sleep(0.3)

        assert len(broadcast_replies) == 1
        assert len(broadcast_replies[0].values) == 3

    loop.run_until_complete(_test())


# =============================================================================
# Tests: stop pool
# =============================================================================


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
            provider_config=None,
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


# =============================================================================
# Tests: preemption handling in ready state
# =============================================================================


def test_pool_forwards_preemption_to_node_and_removes_instance(actor_system):
    system, loop = actor_system
    provider_events: list = []
    start_replies: list = []
    exec_replies: list[ExecuteResult] = []

    async def _test():
        provider_ref = system.spawn(probe_actor(provider_events), "provider-6")
        start_reply_ref = system.spawn(reply_probe(start_replies), "start-reply-6")
        exec_reply_ref = system.spawn(reply_probe(exec_replies), "exec-reply-6")

        pool_ref: ActorRef[PoolMsg] = system.spawn(
            pool_actor(),
            "pool-6",
        )

        spec = _make_spec(nodes=2)
        pool_ref.tell(StartPool(
            spec=spec,
            provider_config=None,
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

        info_0 = _make_instance(0)
        info_1 = _make_instance(1)
        pool_ref.tell(NodeBecameReady(node_id=0, instance=info_0))
        pool_ref.tell(NodeBecameReady(node_id=1, instance=info_1))
        await asyncio.sleep(0.5)

        pool_ref.tell(InstancePreempted(
            instance=info_1,
            reason="spot-interruption",
        ))
        await asyncio.sleep(0.3)

        pool_ref.tell(ExecuteTask(
            fn=lambda: "still-works",
            args=(),
            kwargs={},
            node=None,
            reply_to=exec_reply_ref,
        ))
        await asyncio.sleep(0.3)

        assert len(exec_replies) == 1
        assert exec_replies[0].node_id == 0

        new_info_1 = InstanceMetadata(
            id="i-new-1",
            node=1,
            provider="aws",
            ip="10.0.0.10",
        )
        pool_ref.tell(NodeBecameReady(node_id=1, instance=new_info_1))
        await asyncio.sleep(0.3)

    loop.run_until_complete(_test())


# =============================================================================
# Tests: round-robin execution
# =============================================================================


def test_pool_round_robin_execution(actor_system):
    system, loop = actor_system
    provider_events: list = []
    start_replies: list = []
    exec_replies: list[ExecuteResult] = []

    async def _test():
        provider_ref = system.spawn(probe_actor(provider_events), "provider-7")
        start_reply_ref = system.spawn(reply_probe(start_replies), "start-reply-7")
        exec_reply_ref = system.spawn(reply_probe(exec_replies), "exec-reply-7")

        pool_ref: ActorRef[PoolMsg] = system.spawn(
            pool_actor(),
            "pool-7",
        )

        spec = _make_spec(nodes=2)
        pool_ref.tell(StartPool(
            spec=spec,
            provider_config=None,
            provider_ref=provider_ref,
            reply_to=start_reply_ref,
        ))
        await asyncio.sleep(0.3)

        request_id = [e for e in provider_events if isinstance(e, ClusterRequested)][0].request_id
        pool_ref.tell(ClusterProvisioned(
            request_id=request_id,
            cluster_id="c-rr",
            provider="aws",
        ))
        await asyncio.sleep(0.3)

        pool_ref.tell(NodeBecameReady(node_id=0, instance=_make_instance(0)))
        pool_ref.tell(NodeBecameReady(node_id=1, instance=_make_instance(1)))
        await asyncio.sleep(0.5)

        for _ in range(4):
            pool_ref.tell(ExecuteTask(
                fn=lambda: "ok",
                args=(),
                kwargs={},
                node=None,
                reply_to=exec_reply_ref,
            ))
        await asyncio.sleep(0.5)

        assert len(exec_replies) == 4
        node_ids = [r.node_id for r in exec_replies]
        assert node_ids[0] == 0
        assert node_ids[1] == 1
        assert node_ids[2] == 0
        assert node_ids[3] == 1

    loop.run_until_complete(_test())
