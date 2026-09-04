"""The machines a compute asks for are bought together, not one after another."""

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar, Self

import msgspec
import pytest

from skyward.server.application.machines import Machines
from skyward.server.application.reconciler import Reconciler, Wakeup
from skyward.server.application.runtimes import keypair
from skyward.server.persistence.computes import ComputeStore, GenerationStore, Infrastructure
from skyward.server.persistence.db import connect
from skyward.server.persistence.events import EventStore
from skyward.server.persistence.functions import BlobStore
from skyward.server.persistence.nodes import NodeStore
from skyward.server.persistence.tasks import TaskStore
from skyward.shared.provider import Binding, Machine
from skyward.shared.schemas import ComputeCreate, ComputeSpec, Image, Market, NodeBounds, Offer, Page, ProviderRef, Spec, Worker

pytestmark = pytest.mark.local

NODES = 50
KEY = keypair()[0]


class Gated:
    """A provider whose launches wait until the test lets them all go, counting how many waited at once."""

    kind: ClassVar[str] = "gated"
    credential_fields: ClassVar[tuple[str, ...]] = ()
    offers_ttl: ClassVar[timedelta] = timedelta(minutes=5)

    def __init__(self) -> None:
        self.open = asyncio.Event()
        self.waiting = 0
        self.peak = 0
        self.sold = 0

    @classmethod
    def create(cls, provider_id: str, name: str, credentials: Mapping[str, str], config: Mapping[str, Any]) -> Self:
        return cls()

    async def offers(self) -> AsyncIterator[Offer]:
        yield OFFER

    def allows_cluster_formation(self, spec: ComputeSpec, offer: Offer) -> bool:
        return False

    async def initialize(self, compute_id: str, spec: ComputeSpec, offer: Offer, market: Market, public_key: str) -> Binding:
        return {"compute_id": compute_id}

    async def launch(self, binding: Binding, market: Market, node: str) -> Machine:
        self.waiting += 1
        self.peak = max(self.peak, self.waiting)
        try:
            await self.open.wait()
        finally:
            self.waiting -= 1
        self.sold += 1
        return Machine(id=f"m-{self.sold}", state="pending", user="root", node=node)

    async def machines(self, binding: Binding) -> Mapping[str, Machine]:
        return {}

    async def terminate(self, binding: Binding, machine_ids: tuple[str, ...]) -> None:
        return None

    async def release(self, binding: Binding) -> None:
        return None


OFFER = Offer(
    id="gated-a100",
    provider_id="prv_gated",
    provider_name="gated",
    kind="gated",
    instance_type="gated.a100",
    accelerator="a100",
    accelerator_count=1,
    cpus=8,
    memory_gb=32.0,
    region="nowhere",
    spot_price=1.0,
    on_demand_price=2.0,
    available=NODES,
    fetched_at=datetime.now(UTC),
    expires_at=datetime.now(UTC) + timedelta(hours=1),
    specific={},
)


class Providers:
    def __init__(self, adapter: Gated) -> None:
        self._adapter = adapter

    async def adapter(self, ref: str) -> Gated:
        return self._adapter


class Offers:
    async def list(self, **_: object) -> Page[Offer]:
        return Page(items=(OFFER,))


async def settled(reading: Callable[[], int], for_seconds: float = 0.3) -> bool:
    """Whether the reading stopped moving — every launch that was going to start has started."""
    before = reading()
    await asyncio.sleep(for_seconds)
    return reading() == before


def describe_a_compute_that_asks_for_many_machines() -> None:
    async def it_buys_them_all_at_once(tmp_path: Path) -> None:
        await connect(tmp_path / "skyward.sqlite")
        provider = Gated()
        events = EventStore()
        computes, nodes, blobs = ComputeStore(events), NodeStore(), BlobStore()
        machines = Machines(computes, nodes, Providers(provider), Offers(), blobs, events)  # type: ignore[arg-type]
        requested: list[str] = []
        wake = Wakeup()
        wake.bind(lambda event, **payload: requested.append(payload["node_id"]) if event == "node.requested" else None)
        reconciler = Reconciler(computes, GenerationStore(computes), nodes, TaskStore(computes, nodes, blobs), machines, events, wake)

        spec = ComputeSpec(
            specs=(Spec(provider=ProviderRef(kind="gated"), accelerator="a100", accelerator_count=1),),
            nodes=NodeBounds(initial=NODES),
            image=Image(python="3.13"),
            worker=Worker(concurrency=1, executor="thread"),
        )
        compute, _ = await computes.create(ComputeCreate(spec=spec), idempotency_key="given")
        await reconciler.compute(compute.id)
        requested = list(dict.fromkeys(requested))
        assert len(requested) == NODES

        purchases = [asyncio.create_task(machines.create(compute.id, node_id)) for node_id in requested]
        async with asyncio.timeout(10):
            while provider.waiting < NODES and not await settled(lambda: provider.waiting):
                pass
        provider.open.set()
        await asyncio.gather(*purchases)

        assert provider.peak == NODES, f"only {provider.peak} of {NODES} launches were in flight together"
        assert [node.state for node in await nodes.of(compute.id)] == ["provisioning"] * NODES


class Scarce:
    """One region that never sells, and one that sells exactly once."""

    kind: ClassVar[str] = "scarce"
    credential_fields: ClassVar[tuple[str, ...]] = ()
    offers_ttl: ClassVar[timedelta] = timedelta(minutes=5)

    def __init__(self) -> None:
        self.stock = {"east": 0, "west": 1}
        self.selling = asyncio.Event()

    @classmethod
    def create(cls, provider_id: str, name: str, credentials: Mapping[str, str], config: Mapping[str, Any]) -> Self:
        return cls()

    async def offers(self) -> AsyncIterator[Offer]:
        yield EAST
        yield WEST

    def allows_cluster_formation(self, spec: ComputeSpec, offer: Offer) -> bool:
        return False

    async def initialize(self, compute_id: str, spec: ComputeSpec, offer: Offer, market: Market, public_key: str) -> Binding:
        return {"region": offer.region}

    async def launch(self, binding: Binding, market: Market, node: str) -> Machine:
        region = str(binding["region"])
        if self.stock[region] == 0:
            await asyncio.sleep(0)
            raise RuntimeError(f"{region} has no capacity")
        self.stock[region] -= 1
        self.selling.set()
        await asyncio.sleep(0.05)
        return Machine(id=f"m-{region}", state="pending", user="root", node=node)

    async def machines(self, binding: Binding) -> Mapping[str, Machine]:
        return {}

    async def terminate(self, binding: Binding, machine_ids: tuple[str, ...]) -> None:
        return None

    async def release(self, binding: Binding) -> None:
        return None


EAST = Offer(
    id="scarce-east",
    provider_id="prv_scarce",
    provider_name="scarce",
    kind="scarce",
    instance_type="scarce.a100",
    accelerator="a100",
    accelerator_count=1,
    cpus=8,
    memory_gb=32.0,
    region="east",
    spot_price=1.0,
    on_demand_price=2.0,
    available=1,
    fetched_at=datetime.now(UTC),
    expires_at=datetime.now(UTC) + timedelta(hours=1),
    specific={},
)
WEST = msgspec.structs.replace(EAST, id="scarce-west", region="west", spot_price=1.5, on_demand_price=2.5)


class TwoOffers:
    async def list(self, **_: object) -> Page[Offer]:
        return Page(items=(EAST, WEST))


def describe_two_nodes_refused_by_the_same_region() -> None:
    async def the_one_that_follows_the_move_does_not_hang_when_it_is_refused_again(tmp_path: Path) -> None:
        """The second node finds the compute already moved, follows it, is refused there too, and must keep walking."""
        await connect(tmp_path / "skyward.sqlite")
        provider = Scarce()
        events = EventStore()
        computes, nodes, blobs = ComputeStore(events), NodeStore(), BlobStore()
        machines = Machines(computes, nodes, Providers(provider), TwoOffers(), blobs, events)  # type: ignore[arg-type]

        spec = ComputeSpec(
            specs=(Spec(provider=ProviderRef(kind="scarce"), accelerator="a100", accelerator_count=1),),
            nodes=NodeBounds(initial=2),
            image=Image(python="3.13"),
            worker=Worker(concurrency=1, executor="thread"),
        )
        compute, _ = await computes.create(ComputeCreate(spec=spec), idempotency_key="given")
        east = Infrastructure(provider_id="prv_scarce", offer_id=EAST.id, offer=EAST, binding={"region": "east"}, private_key=KEY, markets=("spot",))
        await computes.bind(compute.id, east)
        compute = await computes.get(compute.id)

        async with asyncio.timeout(5):
            first, second = await asyncio.gather(
                machines._place(provider, compute, east, "n1"),
                machines._place(provider, compute, east, "n2"),
                return_exceptions=True,
            )

        outcomes = sorted((first, second), key=lambda outcome: isinstance(outcome, BaseException))
        placed, refused = outcomes
        assert isinstance(placed, tuple) and placed[1].id == "m-west"
        assert isinstance(refused, ExceptionGroup)
        assert {str(failure) for failure in refused.exceptions} == {"east has no capacity", "west has no capacity"}
