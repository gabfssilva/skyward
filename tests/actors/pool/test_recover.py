"""RecoverPool spawns node actors that Adopt with persisted ranks."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from casty import ActorContext, ActorSystem, Behavior, Behaviors

from skyward.actors.node.messages import Adopt
from skyward.actors.pool.actor import pool_actor
from skyward.actors.pool.messages import RecoverPool

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


def _spec() -> MagicMock:
    spec = MagicMock()
    spec.autoscale_idle_timeout = 60.0
    spec.ssh_timeout = 30
    spec.ssh_retry_interval = 2
    spec.provision_timeout = 300
    spec.bootstrap_timeout = 300
    spec.cluster = True
    return spec


async def test_recover_pool_adopts_with_persisted_ranks(monkeypatch):
    records: list[tuple[int, object]] = []

    def fake_node_actor(node_id: int, pool, **_kwargs):  # noqa: ANN001
        async def receive(ctx: ActorContext, msg: object) -> Behavior:
            records.append((node_id, msg))
            return Behaviors.same()
        return Behaviors.receive(receive)

    monkeypatch.setattr("skyward.actors.pool.actor.node_actor", fake_node_actor)

    inst0 = MagicMock(name="inst0")
    inst1 = MagicMock(name="inst1")
    cluster = MagicMock()
    cluster.id = "c-1"

    async with ActorSystem("pool-recover") as system:
        ref = system.spawn(pool_actor(), "pool")
        ref.tell(RecoverPool(
            spec=_spec(), provider=MagicMock(), cluster=cluster,
            instances=(inst0, inst1), reply_to=MagicMock(), node_ids=(2, 5),
        ))
        await asyncio.sleep(0.2)

    # persisted ranks used (not 0,1 enumeration); every node got Adopt, not Provision
    by_rank = {nid: msg for nid, msg in records}
    assert set(by_rank) == {2, 5}
    assert all(isinstance(msg, Adopt) for msg in by_rank.values())
    assert by_rank[2].instance is inst0
    assert by_rank[5].instance is inst1


async def test_recover_pool_falls_back_to_enumeration(monkeypatch):
    records: list[tuple[int, object]] = []

    def fake_node_actor(node_id: int, pool, **_kwargs):  # noqa: ANN001
        async def receive(ctx: ActorContext, msg: object) -> Behavior:
            records.append((node_id, msg))
            return Behaviors.same()
        return Behaviors.receive(receive)

    monkeypatch.setattr("skyward.actors.pool.actor.node_actor", fake_node_actor)

    async with ActorSystem("pool-recover-enum") as system:
        ref = system.spawn(pool_actor(), "pool")
        ref.tell(RecoverPool(
            spec=_spec(), provider=MagicMock(), cluster=MagicMock(),
            instances=(MagicMock(), MagicMock()), reply_to=MagicMock(),
        ))
        await asyncio.sleep(0.2)

    assert {nid for nid, _ in records} == {0, 1}
