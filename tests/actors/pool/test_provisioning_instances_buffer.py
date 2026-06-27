"""A node that becomes ready during ``provisioning_instances`` must still join.

Reproduces the fixed-count hang: when the provider fulfils the instance
request across multiple rounds (partial provision + retry), nodes spawned
in an early round can reach ``NodeBecameReady`` while the pool is still
accumulating instances in ``provisioning_instances``. That event must not
be dropped — otherwise the node never receives ``JoinCluster``, never
activates, and a ``nodes=int`` pool (whose readiness threshold is the full
desired count) waits forever.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from casty import ActorContext, Behavior, Behaviors

from skyward.actors.messages import NodeBecameReady
from skyward.actors.node.messages import JoinCluster
from skyward.actors.pool.actor import pool_actor
from skyward.actors.pool.messages import StartPool
from skyward.api.spec import Nodes, PoolSpec
from skyward.core.model import Cluster

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class _FakeClusterClient:
    """Stand-in for casty's ``ClusterClient`` — no real network."""

    def __init__(self, **_kwargs: Any) -> None: ...

    async def __aenter__(self) -> _FakeClusterClient:
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False


def _spec() -> PoolSpec:
    return PoolSpec(
        nodes=Nodes(desired=2),
        accelerator=None,
        region="test",
        cluster=False,
        provision_retry_delay=0.0,
        max_provision_attempts=5,
    )


def _cluster(spec: PoolSpec) -> Cluster[Any]:
    return Cluster(
        id="c1",
        status="ready",
        spec=spec,
        offer=MagicMock(),
        ssh_key_path="/key",
        ssh_user="root",
        use_sudo=False,
        shutdown_command="shutdown",
        specific=MagicMock(),
        instances=(),
        prebaked=False,
        mount_plan=None,
    )


async def test_node_ready_during_provisioning_instances_still_joins(monkeypatch):
    records: dict[int, list[object]] = {}

    def fake_node_actor(node_id: int, pool: Any, **_kwargs: Any) -> Behavior:
        async def receive(ctx: ActorContext, msg: object) -> Behavior:
            records.setdefault(node_id, []).append(msg)
            return Behaviors.same()
        return Behaviors.receive(receive)

    monkeypatch.setattr("skyward.actors.pool.actor.node_actor", fake_node_actor)
    monkeypatch.setattr("skyward.actors.pool.actor.ClusterClient", _FakeClusterClient)
    monkeypatch.setattr(
        "skyward.actors.pool.actor._build_pool_info_json",
        lambda *_a, **_k: "{}",
    )
    monkeypatch.setattr("skyward.infra.tls.ensure_ca", lambda: object())
    monkeypatch.setattr("skyward.infra.tls.issue_client_config", lambda _ca: None)

    spec = _spec()
    cluster = _cluster(spec)

    gate = asyncio.Event()
    calls = {"n": 0}

    async def fake_provision(_cluster: Any, _count: int) -> tuple[Any, tuple[Any, ...]]:
        calls["n"] += 1
        if calls["n"] == 1:
            return cluster, (MagicMock(name="inst0"),)
        await gate.wait()  # hold batch 2 until node 0 has gone ready
        return cluster, (MagicMock(name="inst1"),)

    provider = MagicMock()
    provider.prepare = _async_return(cluster)
    provider.provision = fake_provision

    from casty import ActorSystem

    async with ActorSystem("pool-pi-buffer") as system:
        ref = system.spawn(pool_actor(), "pool")
        ref.tell(StartPool(
            spec=spec, provider_config=MagicMock(), provider=provider,
            offers=(MagicMock(),), reply_to=MagicMock(),
        ))

        # Let batch 1 land: node 0 spawned, pool parked in provisioning_instances
        # with the retry (batch 2) blocked on the gate.
        await asyncio.sleep(0.2)

        # Node 0 reaches readiness while the pool is still gathering instances.
        ref.tell(NodeBecameReady(
            node_id=0, instance=MagicMock(),
            local_port=5000, private_ip="10.0.0.1", casty_port=25520,
        ))
        await asyncio.sleep(0.1)

        gate.set()  # release batch 2 → pool transitions to provisioning
        await asyncio.sleep(0.3)

    node0_msgs = records.get(0, [])
    assert any(isinstance(m, JoinCluster) for m in node0_msgs), (
        "node 0 became ready during provisioning_instances but never received "
        f"JoinCluster — it would never activate. Got: {[type(m).__name__ for m in node0_msgs]}"
    )


def _async_return(value: Any):
    async def _fn(*_a: Any, **_k: Any) -> Any:
        return value
    return _fn
