from __future__ import annotations

import asyncio

import pytest
from casty import ActorContext, Behavior, Behaviors
from casty.sharding import ClusteredActorSystem

from skyward.actors.messages import (
    InstanceBootstrapped,
    InstanceMetadata,
    InstancePreempted,
    InstanceProvisioned,
    InstanceRequested,
    NodeBecameReady,
    NodeMsg,
    Provision,
)
from skyward.actors.node import node_actor
from tests.conftest import get_free_port


def probe_actor(results: list) -> Behavior:
    async def receive(ctx: ActorContext, msg) -> Behavior:
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
        system = ClusteredActorSystem(name="test-node", host="127.0.0.1", port=get_free_port(), node_id="node-test-0")
        await system.__aenter__()
        return system

    system = event_loop.run_until_complete(_start())
    yield system, event_loop
    event_loop.run_until_complete(system.__aexit__(None, None, None))


def test_node_sends_instance_requested_on_start(actor_system):
    system, loop = actor_system
    provider_collected: list = []
    pool_collected: list = []

    async def _test():
        provider_probe_ref = system.spawn(probe_actor(provider_collected), "provider-probe-1")
        pool_probe_ref = system.spawn(probe_actor(pool_collected), "pool-probe-1")

        node_ref = system.spawn(
            node_actor(
                node_id=0,
                cluster_id="c-test",
                provider="aws",
                provider_ref=provider_probe_ref,
                pool_ref=pool_probe_ref,
            ),
            "node-0",
        )

        node_ref.tell(Provision(cluster_id="c-test", provider="aws"))
        await asyncio.sleep(0.5)

        assert len(provider_collected) >= 1
        assert isinstance(provider_collected[0], InstanceRequested)
        assert provider_collected[0].node_id == 0
        assert provider_collected[0].cluster_id == "c-test"

    loop.run_until_complete(_test())


def test_node_transitions_to_ready_on_bootstrap(actor_system):
    system, loop = actor_system
    provider_collected: list = []
    pool_collected: list = []

    async def _test():
        provider_probe_ref = system.spawn(probe_actor(provider_collected), "provider-probe-2")
        pool_probe_ref = system.spawn(probe_actor(pool_collected), "pool-probe-2")

        node_ref = system.spawn(
            node_actor(
                node_id=1,
                cluster_id="c-test-2",
                provider="aws",
                provider_ref=provider_probe_ref,
                pool_ref=pool_probe_ref,
            ),
            "node-1",
        )

        node_ref.tell(Provision(cluster_id="c-test-2", provider="aws"))
        await asyncio.sleep(0.3)

        info = InstanceMetadata(id="i-abc", node=1, provider="aws", ip="10.0.0.1")
        node_ref.tell(InstanceProvisioned(request_id="node-1-abc", instance=info))
        await asyncio.sleep(0.3)

        node_ref.tell(InstanceBootstrapped(instance=info))
        await asyncio.sleep(0.3)

        pool_ready_events = [e for e in pool_collected if isinstance(e, NodeBecameReady)]
        assert len(pool_ready_events) == 1
        assert pool_ready_events[0].node_id == 1

    loop.run_until_complete(_test())


def test_node_replaces_on_preemption(actor_system):
    system, loop = actor_system
    provider_collected: list = []
    pool_collected: list = []

    async def _test():
        provider_probe_ref = system.spawn(probe_actor(provider_collected), "provider-probe-3")
        pool_probe_ref = system.spawn(probe_actor(pool_collected), "pool-probe-3")

        node_ref = system.spawn(
            node_actor(
                node_id=2,
                cluster_id="c-test-3",
                provider="aws",
                provider_ref=provider_probe_ref,
                pool_ref=pool_probe_ref,
            ),
            "node-2",
        )

        node_ref.tell(Provision(cluster_id="c-test-3", provider="aws"))
        await asyncio.sleep(0.3)

        info = InstanceMetadata(id="i-old", node=2, provider="aws", ip="10.0.0.2")
        node_ref.tell(InstanceProvisioned(request_id="node-2-abc", instance=info))
        await asyncio.sleep(0.2)
        node_ref.tell(InstanceBootstrapped(instance=info))
        await asyncio.sleep(0.3)

        node_ref.tell(InstancePreempted(instance=info, reason="spot-interruption"))
        await asyncio.sleep(0.5)

        replace_requests = [e for e in provider_collected if isinstance(e, InstanceRequested) and e.replacing is not None]
        assert len(replace_requests) == 1
        assert replace_requests[0].replacing == "i-old"

    loop.run_until_complete(_test())
