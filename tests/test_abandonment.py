"""A compute nobody owns is not a compute to keep buying machines for.

The tick re-offers every live compute to the reconciler, and the store is shared
between runs — so a compute left behind by a killed script would be provisioned
again on every tick, by every later run, forever. The lease is what tells a
newborn from a corpse: reconciliation only drives a compute forward while someone
holds it, and past the grace an ownerless ``delete_on_exit`` compute is torn down
instead.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from skyward.server.application.reconciler import ABANDON_SECONDS, Reconciler, Wakeup
from skyward.server.persistence.computes import ComputeStore, GenerationStore
from skyward.server.persistence.db import connect
from skyward.server.persistence.events import EventStore
from skyward.server.persistence.functions import BlobStore
from skyward.server.persistence.nodes import NodeStore
from skyward.server.persistence.store import now
from skyward.server.persistence.tables import ComputeRow
from skyward.server.persistence.tasks import TaskStore
from skyward.shared.schemas import (
    Compute,
    ComputeCreate,
    ComputeSpec,
    Image,
    LeaseClaim,
    Node,
    NodeBounds,
    ProviderRef,
    Spec,
)

pytestmark = pytest.mark.unit


class NoCloud:
    """A ``Machines`` that panics if the reconciler ever reaches for a provider."""

    async def resolve(self, compute: Compute, nodes: tuple[Node, ...]) -> None:
        return None

    async def terminate(self, compute_id: str, node_id: str) -> None:
        raise AssertionError("nothing here should touch a machine")

    async def release(self, compute_id: str) -> None:
        return None


class World:
    def __init__(self) -> None:
        self.computes = ComputeStore()
        self.nodes = NodeStore()
        self.events = EventStore()
        self.tasks = TaskStore(self.computes, self.nodes, BlobStore())
        self.reconciler = Reconciler(
            computes=self.computes,
            generations=GenerationStore(self.computes),
            nodes=self.nodes,
            tasks=self.tasks,
            machines=NoCloud(),  # type: ignore[arg-type]
            events=self.events,
            wake=Wakeup(),
        )

    async def compute(self, delete_on_exit: bool) -> Compute:
        spec = ComputeSpec(
            specs=(Spec(provider=ProviderRef(kind="container"), cpus=1, memory_gb=1),),
            nodes=NodeBounds(desired=2),
            image=Image(python="3.13", skyward="local"),
            delete_on_exit=delete_on_exit,
        )
        created, _ = await self.computes.create(ComputeCreate(spec=spec, name=None), idempotency_key=created_key(spec))
        return created

    async def aged(self, compute_id: str) -> None:
        await ComputeRow.update(
            {ComputeRow.created_at: now() - timedelta(seconds=ABANDON_SECONDS + 1)},
        ).where(ComputeRow.id == compute_id).run()


def created_key(spec: ComputeSpec) -> str:
    return f"k_{id(spec)}"


@pytest.fixture
async def world(tmp_path: Path) -> World:
    await connect(tmp_path / "skyward.sqlite")
    return World()


async def test_an_abandoned_compute_with_delete_on_exit_is_torn_down(world: World):
    created = await world.compute(delete_on_exit=True)
    await world.aged(created.id)

    await world.reconciler.compute(created.id)

    compute = await world.computes.get(created.id)
    assert compute.spec.desired == "deleted", "no live lease past the grace means nobody is coming back"
    assert await world.nodes.of(created.id) == (), "not one machine was requested for it"


async def test_an_abandoned_compute_without_delete_on_exit_just_sits(world: World):
    created = await world.compute(delete_on_exit=False)
    await world.aged(created.id)

    await world.reconciler.compute(created.id)

    compute = await world.computes.get(created.id)
    assert compute.spec.desired == "running", "an ownerless keepable compute is left for someone to attach"
    assert await world.nodes.of(created.id) == (), "but nothing provisions on its behalf"


async def test_a_newborn_without_a_lease_still_provisions(world: World):
    created = await world.compute(delete_on_exit=True)

    await world.reconciler.compute(created.id)

    assert len(await world.nodes.of(created.id)) == 2, "the grace covers the moment between create and claim"


async def test_a_leased_compute_provisions_past_the_grace(world: World):
    created = await world.compute(delete_on_exit=True)
    await world.aged(created.id)
    await world.computes.claim_lease(created.id, LeaseClaim(owner="sdk_test", ttl_seconds=60))

    await world.reconciler.compute(created.id)

    assert len(await world.nodes.of(created.id)) == 2, "a live lease is a live owner"
