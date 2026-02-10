from __future__ import annotations

import asyncio

import pytest
from casty import ActorContext, ActorRef, Behavior, Behaviors
from casty.sharding import ClusteredActorSystem

from skyward.actors.provider import BootstrapDone, InstanceReady, ProviderMsg, _ProvisioningDone
from skyward.image import Image
from skyward.messages import ClusterProvisioned, ClusterRequested, ShutdownRequested
from skyward.spec import PoolSpec
from tests.conftest import get_free_port


def probe_actor(results: list) -> Behavior:
    async def receive(ctx, msg):
        results.append(msg)
        return Behaviors.same()
    return Behaviors.receive(receive)


def mock_provider_actor(
    pool_ref: ActorRef,
) -> Behavior[ProviderMsg]:
    """A fake provider that immediately provisions."""

    def idle() -> Behavior[ProviderMsg]:
        async def receive(ctx: ActorContext[ProviderMsg], msg: ProviderMsg) -> Behavior[ProviderMsg]:
            match msg:
                case ClusterRequested(request_id=rid, provider=prov):
                    event = ClusterProvisioned(request_id=rid, cluster_id=f"c-{rid}", provider=prov)
                    pool_ref.tell(event)
                    return active(cluster_id=f"c-{rid}")
                case _:
                    return Behaviors.same()

        return Behaviors.receive(receive)

    def active(cluster_id: str) -> Behavior[ProviderMsg]:
        async def receive(ctx: ActorContext[ProviderMsg], msg: ProviderMsg) -> Behavior[ProviderMsg]:
            match msg:
                case ShutdownRequested():
                    return Behaviors.stopped()
                case _:
                    return Behaviors.same()

        return Behaviors.receive(receive)

    return idle()


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def actor_system(event_loop):
    async def _start():
        system = ClusteredActorSystem(name="test-provider", host="127.0.0.1", port=get_free_port(), node_id="prov-0")
        await system.__aenter__()
        return system

    system = event_loop.run_until_complete(_start())
    yield system, event_loop
    event_loop.run_until_complete(system.__aexit__(None, None, None))


def test_provider_handles_cluster_requested(actor_system):
    system, loop = actor_system
    collected: list = []

    async def _test():
        probe_ref = system.spawn(probe_actor(collected), "probe-prov")
        prov_ref = system.spawn(mock_provider_actor(pool_ref=probe_ref), "mock-provider")

        spec = PoolSpec(nodes=1, accelerator="A100", region="us-east-1", image=Image())
        prov_ref.tell(ClusterRequested(request_id="r-1", provider="aws", spec=spec))
        await asyncio.sleep(0.5)

        assert len(collected) == 1
        assert isinstance(collected[0], ClusterProvisioned)
        assert collected[0].cluster_id == "c-r-1"

    loop.run_until_complete(_test())


def test_provider_message_types_importable():
    ready = InstanceReady(
        instance_id="i-123",
        node_id=0,
        ip="1.2.3.4",
        private_ip=None,
        ssh_port=22,
        spot=False,
        metadata={},
    )
    assert ready.instance_id == "i-123"

    from skyward.messages import InstanceMetadata

    fake_instance = InstanceMetadata(id="i-123", node=0, provider="aws", ip="1.2.3.4")
    done = BootstrapDone(instance=fake_instance, success=True)
    assert done.success is True
    assert done.error is None

    done_fail = BootstrapDone(instance=fake_instance, success=False, error="timeout")
    assert done_fail.error == "timeout"
