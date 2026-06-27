"""Node reattach: Adopt enters at connecting and skips bootstrap/worker."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock

import pytest
from casty import ActorContext, ActorSystem, Behavior, Behaviors

from skyward.actors.messages import NodeBecameReady, NodeConnected
from skyward.actors.node.actor import node_actor
from skyward.actors.node.messages import Adopt
from skyward.api.model import Instance, InstanceType, Offer
from skyward.infra.ssh_actor import ForwardPort, PortForwarded

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


def _instance() -> Instance:
    itype = InstanceType(
        name="t", accelerator=None, vcpus=1, memory_gb=1,
        architecture="x86_64", specific=None,
    )
    offer = Offer(
        id="o", instance_type=itype, spot_price=1.0, on_demand_price=2.0,
        billing_unit="hour", specific=None,
    )
    return Instance(id="i-0", status="ready", offer=offer, ip="1.2.3.4", private_ip="10.0.0.1")


def _fake_transport(**_kwargs: object) -> Behavior:
    async def receive(ctx: ActorContext, msg: object) -> Behavior:
        if isinstance(msg, ForwardPort):
            msg.reply_to.tell(PortForwarded(local_port=12345))
        return Behaviors.same()
    return Behaviors.receive(receive)


def _probe(events: list[object]) -> Behavior:
    async def receive(ctx: ActorContext, msg: object) -> Behavior:
        events.append(msg)
        return Behaviors.same()
    return Behaviors.receive(receive)


async def test_adopt_skips_bootstrap_and_worker(monkeypatch):
    calls = {"bootstrap": 0, "worker": 0}

    async def _no_bootstrap(*_a, **_k):
        calls["bootstrap"] += 1

    async def _no_worker(*_a, **_k):
        calls["worker"] += 1
        return (0, "")

    monkeypatch.setattr("skyward.actors.node.actor.ssh_transport", _fake_transport)
    monkeypatch.setattr("skyward.actors.node.actor.run_bootstrap", _no_bootstrap)
    monkeypatch.setattr("skyward.actors.node.actor.do_start_worker", _no_worker)

    cluster = MagicMock()
    cluster.spec.provider = "container"
    cluster.ssh_user = "root"
    cluster.ssh_key_path = "/k"
    inst = _instance()
    # Adopt refreshes coordinates via one get_instance (handles port remapping),
    # then connects and SKIPS bootstrap/worker.
    provider = MagicMock()
    provider.get_instance = AsyncMock(return_value=(cluster, replace(inst, status="provisioned")))

    async with ActorSystem("node-adopt") as system:
        events: list[object] = []
        pool = system.spawn(_probe(events), "pool")
        node = system.spawn(node_actor(node_id=0, pool=pool, _skip_monitor=True), "node")
        node.tell(Adopt(cluster=cluster, provider=provider, instance=inst))
        await asyncio.sleep(0.4)

    types = [type(e).__name__ for e in events]
    assert "NodeConnected" in types
    assert "NodeBecameReady" in types
    became_ready = next(e for e in events if isinstance(e, NodeBecameReady))
    assert became_ready.node_id == 0
    assert became_ready.local_port == 12345
    assert calls["bootstrap"] == 0
    assert calls["worker"] == 0
    assert provider.get_instance.await_count == 1
    connected = next(e for e in events if isinstance(e, NodeConnected))
    assert connected.node_id == 0
