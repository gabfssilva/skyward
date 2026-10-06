"""The machines a compute asks for are bought together, not one after another."""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar, Self

import msgspec
import pytest

from skyward.server.application.machines import REFUSED_FIRST, REFUSED_MAX, Machines
from skyward.server.application.reconciler import Reconciler, Wakeup
from skyward.server.application.runtimes import keypair
from skyward.server.persistence.computes import ComputeStore, GenerationStore, Infrastructure
from skyward.server.persistence.db import connect
from skyward.server.persistence.events import EventStore
from skyward.server.persistence.functions import BlobStore
from skyward.server.persistence.nodes import NodeStore
from skyward.server.persistence.store import now
from skyward.server.persistence.tasks import TaskStore
from skyward.shared.provider import Binding, Machine
from skyward.shared.schemas import Compute, ComputeCreate, ComputeSpec, Image, Market, NodeBounds, Offer, Page, ProviderRef, Spec, Worker

pytestmark = pytest.mark.local

NODES = 50
KEY = keypair()[0]


class Gated:
    """A provider whose launches wait until all of them are waiting at once, counting how many did."""

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
        if self.waiting == NODES:
            self.open.set()
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


def describe_a_compute_that_asks_for_many_machines() -> None:
    async def it_buys_them_all_at_once(tmp_path: Path) -> None:
        await connect(tmp_path / "skyward.sqlite")
        provider = Gated()
        events = EventStore()
        nodes, blobs = NodeStore(), BlobStore()
        computes = ComputeStore(events, nodes)
        machines = Machines(computes, nodes, Providers(provider), Offers(), blobs, events)  # type: ignore[arg-type]
        requested: list[str] = []
        wake = Wakeup()
        wake.bind(lambda event, **payload: requested.append(payload["node_id"]) if event == "node.requested" else None)
        reconciler = Reconciler(computes, GenerationStore(computes), nodes, TaskStore(computes, nodes, blobs, events), machines, events, wake)

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
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(provider.open.wait(), timeout=10)
        provider.open.set()
        await asyncio.gather(*purchases)

        assert provider.peak == NODES, f"only {provider.peak} of {NODES} launches were in flight together"
        assert [node.state for node in await nodes.of(compute.id)] == ["provisioning"] * NODES


class Scarce:
    """One offer that never sells, and two that sell exactly once: the same type elsewhere, and a bigger type in the same place."""

    kind: ClassVar[str] = "scarce"
    credential_fields: ClassVar[tuple[str, ...]] = ()
    offers_ttl: ClassVar[timedelta] = timedelta(minutes=5)

    def __init__(self) -> None:
        self.stock = {EAST.id: 0, BIGGER.id: 1, WEST.id: 1}
        self.selling = asyncio.Event()

    @classmethod
    def create(cls, provider_id: str, name: str, credentials: Mapping[str, str], config: Mapping[str, Any]) -> Self:
        return cls()

    async def offers(self) -> AsyncIterator[Offer]:
        yield EAST
        yield BIGGER
        yield WEST

    def allows_cluster_formation(self, spec: ComputeSpec, offer: Offer) -> bool:
        return False

    async def initialize(self, compute_id: str, spec: ComputeSpec, offer: Offer, market: Market, public_key: str) -> Binding:
        return {"region": offer.region, "offer": offer.id}

    async def launch(self, binding: Binding, market: Market, node: str) -> Machine:
        region, offer = str(binding["region"]), str(binding["offer"])
        if self.stock[offer] == 0:
            await asyncio.sleep(0)
            raise RuntimeError(f"{region} has no capacity")
        self.stock[offer] -= 1
        self.selling.set()
        await asyncio.sleep(0.05)
        return Machine(id=f"m-{offer.removeprefix('scarce-')}", state="pending", user="root", node=node)

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
BIGGER = msgspec.structs.replace(EAST, id="scarce-east-bigger", instance_type="scarce.a100x", spot_price=1.2, on_demand_price=2.2)
WEST = msgspec.structs.replace(EAST, id="scarce-west", region="west", spot_price=1.5, on_demand_price=2.5)


class ScarceOffers:
    async def list(self, **_: object) -> Page[Offer]:
        return Page(items=(EAST, BIGGER, WEST))


async def bound_east(tmp_path: Path, provider: Scarce, **allowed: bool) -> tuple[Machines, ComputeStore, Compute, Infrastructure]:
    await connect(tmp_path / "skyward.sqlite")
    events = EventStore()
    nodes, blobs = NodeStore(), BlobStore()
    computes = ComputeStore(events, nodes)
    machines = Machines(computes, nodes, Providers(provider), ScarceOffers(), blobs, events)  # type: ignore[arg-type]

    spec = ComputeSpec(
        specs=(Spec(provider=ProviderRef(kind="scarce"), accelerator="a100", accelerator_count=1),),
        nodes=NodeBounds(initial=2),
        image=Image(python="3.13"),
        worker=Worker(concurrency=1, executor="thread"),
    )
    compute, _ = await computes.create(ComputeCreate(spec=spec), idempotency_key="given")
    east = Infrastructure(
        provider_id="prv_scarce", offer_id=EAST.id, offer=EAST, binding={"region": "east", "offer": EAST.id, **allowed}, private_key=KEY, markets=("spot",)
    )
    await computes.bind(compute.id, east)
    return machines, computes, await computes.get(compute.id), east


async def peer(machines: Machines, compute: Compute) -> None:
    """A machine of the compute already alive where it was bound."""
    node = await machines._nodes.request(compute.id, compute.generation)
    await machines._nodes.launched(node.id, Machine(id="m-peer", state="running", user="root", node=node.id), EAST, "spot")


def describe_a_node_refused_where_the_compute_was_bound() -> None:
    def with_a_peer_alive() -> None:
        async def it_is_refused_rather_than_bought_somewhere_else(tmp_path: Path) -> None:
            provider = Scarce()
            machines, computes, compute, east = await bound_east(tmp_path, provider)
            await peer(machines, compute)

            with pytest.raises(ExceptionGroup) as refused:
                await machines._place(provider, compute, east, "n1")

            assert [str(failure) for failure in refused.value.exceptions] == ["east has no capacity"]
            assert provider.stock == {EAST.id: 0, BIGGER.id: 1, WEST.id: 1}
            assert (await computes.infrastructure(compute.id)).offer_id == EAST.id

        async def it_is_bought_in_another_place_under_a_multi_subnet_binding_and_is_the_same_instance(tmp_path: Path) -> None:
            provider = Scarce()
            machines, computes, compute, east = await bound_east(tmp_path, provider, multi_subnet=True)
            await peer(machines, compute)

            _, machine, _ = await machines._place(provider, compute, east, "n1")

            assert machine.id == "m-west"
            assert provider.stock[BIGGER.id] == 1, "the cheaper offer is another instance, which this binding did not allow"
            assert (await computes.infrastructure(compute.id)).offer_id == WEST.id

        async def it_is_bought_as_another_instance_under_a_heterogeneous_binding_and_stays_where_it_is(tmp_path: Path) -> None:
            provider = Scarce()
            machines, computes, compute, east = await bound_east(tmp_path, provider, heterogeneous_instances=True)
            await peer(machines, compute)
            provider.stock[BIGGER.id] = 0

            with pytest.raises(ExceptionGroup):
                await machines._place(provider, compute, east, "n1")
            assert provider.stock[WEST.id] == 1, "the west sells, and it is another place, which this binding did not allow"

            provider.stock[BIGGER.id] = 1
            _, machine, _ = await machines._place(provider, compute, east, "n1")

            assert machine.id == "m-east-bigger"
            assert (await computes.infrastructure(compute.id)).offer_id == BIGGER.id

    def with_no_machine_alive() -> None:
        async def it_is_bought_wherever_sells_because_there_is_no_peer_to_be_apart_from(tmp_path: Path) -> None:
            provider = Scarce()
            machines, computes, compute, east = await bound_east(tmp_path, provider)

            _, machine, _ = await machines._place(provider, compute, east, "n1")

            assert machine.id == "m-east-bigger"
            assert (await computes.infrastructure(compute.id)).offer_id == BIGGER.id


def describe_two_nodes_refused_by_the_same_region() -> None:
    async def the_one_that_follows_the_move_does_not_hang_when_it_is_refused_again(tmp_path: Path) -> None:
        """The second node finds the compute already moved, follows it, is refused there too, and must keep walking."""
        provider = Scarce()
        machines, _, compute, east = await bound_east(tmp_path, provider, multi_subnet=True)
        provider.stock[BIGGER.id] = 0

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


class Refusing:
    """One region whose launches are refused until the test lets it sell, counting every launch it was asked for."""

    kind: ClassVar[str] = "refusing"
    credential_fields: ClassVar[tuple[str, ...]] = ()
    offers_ttl: ClassVar[timedelta] = timedelta(minutes=5)

    def __init__(self) -> None:
        self.selling = False
        self.launches = 0

    @classmethod
    def create(cls, provider_id: str, name: str, credentials: Mapping[str, str], config: Mapping[str, Any]) -> Self:
        return cls()

    async def offers(self) -> AsyncIterator[Offer]:
        yield REFUSING

    def allows_cluster_formation(self, spec: ComputeSpec, offer: Offer) -> bool:
        return False

    async def initialize(self, compute_id: str, spec: ComputeSpec, offer: Offer, market: Market, public_key: str) -> Binding:
        return {"region": offer.region}

    async def launch(self, binding: Binding, market: Market, node: str) -> Machine:
        self.launches += 1
        if not self.selling:
            raise RuntimeError("no capacity")
        return Machine(id=f"m-{self.launches}", state="pending", user="root", node=node)

    async def machines(self, binding: Binding) -> Mapping[str, Machine]:
        return {}

    async def terminate(self, binding: Binding, machine_ids: tuple[str, ...]) -> None:
        return None

    async def release(self, binding: Binding) -> None:
        return None


REFUSING = msgspec.structs.replace(EAST, id="refusing-a100", provider_id="prv_refusing", provider_name="refusing", kind="refusing")


class OneOffer:
    async def list(self, **_: object) -> Page[Offer]:
        return Page(items=(REFUSING,))


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Refused:
    """A bound compute with requested rows, bought from a provider that refuses until told otherwise."""

    def __init__(self, machines: Machines, provider: Refusing, compute_id: str, rows: list[str], clock: Clock) -> None:
        self.machines = machines
        self.provider = provider
        self.compute_id = compute_id
        self.rows = rows
        self.clock = clock

    async def attempt(self) -> bool:
        """Ask for the next requested row; whether the provider was called."""
        before = self.provider.launches
        try:
            await self.machines.create(self.compute_id, self.rows[0])
        except ExceptionGroup:
            pass
        else:
            if self.provider.launches > before:
                self.rows.pop(0)
        return self.provider.launches > before

    async def refused(self) -> None:
        assert await self.attempt()

    async def window(self) -> float:
        """How long, from now, the compute waits before the provider is called again."""
        start = self.clock.now
        waited = 0.0
        while not await self.attempt():
            waited += 1.0
            self.clock.now = start + waited
        return waited


@pytest.fixture
async def refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Refused:
    await connect(tmp_path / "skyward.sqlite")
    clock = Clock()
    monkeypatch.setattr("skyward.server.application.machines.monotonic", clock)
    provider = Refusing()
    events = EventStore()
    nodes, blobs = NodeStore(), BlobStore()
    computes = ComputeStore(events, nodes)
    machines = Machines(computes, nodes, Providers(provider), OneOffer(), blobs, events)  # type: ignore[arg-type]
    requested: list[str] = []
    wake = Wakeup()
    wake.bind(lambda event, **payload: requested.append(payload["node_id"]) if event == "node.requested" else None)
    reconciler = Reconciler(computes, GenerationStore(computes), nodes, TaskStore(computes, nodes, blobs, events), machines, events, wake)

    spec = ComputeSpec(
        specs=(Spec(provider=ProviderRef(kind="refusing"), accelerator="a100", accelerator_count=1),),
        nodes=NodeBounds(initial=4),
        image=Image(python="3.13"),
        worker=Worker(concurrency=1, executor="thread"),
    )
    compute, _ = await computes.create(ComputeCreate(spec=spec), idempotency_key="given")
    await reconciler.compute(compute.id)
    infrastructure = Infrastructure(
        provider_id="prv_refusing", offer_id=REFUSING.id, offer=REFUSING, binding={"region": "east"}, private_key=KEY, markets=("spot",)
    )
    await computes.bind(compute.id, infrastructure)
    return Refused(machines, provider, compute.id, list(dict.fromkeys(requested)), clock)


class Unreachable(Refusing):
    """A provider whose account cannot even be opened — the cloud's API is not answering at all."""

    kind: ClassVar[str] = "unreachable"

    async def initialize(self, compute_id: str, spec: ComputeSpec, offer: Offer, market: Market, public_key: str) -> Binding:
        raise RuntimeError("failed to connect to the docker API")


def describe_a_compute_whose_provider_cannot_be_reached() -> None:
    async def it_says_so_on_the_compute_rather_than_only_in_the_log(tmp_path: Path) -> None:
        """A launch is not where every purchase fails: one that fails before it is still a machine nobody bought."""
        await connect(tmp_path / "skyward.sqlite")
        events = EventStore()
        nodes, blobs = NodeStore(), BlobStore()
        computes = ComputeStore(events, nodes)
        machines = Machines(computes, nodes, Providers(Unreachable()), OneOffer(), blobs, events)  # type: ignore[arg-type]
        spec = ComputeSpec(
            specs=(Spec(provider=ProviderRef(kind="unreachable"), accelerator="a100"),),
            nodes=NodeBounds(initial=1),
            image=Image(python="3.13"),
        )
        compute, _ = await computes.create(ComputeCreate(spec=spec), idempotency_key="unreachable")
        node = await nodes.request(compute.id, compute.generation)

        with pytest.raises(Exception, match="docker API"):
            await machines.create(compute.id, node.id)

        placement = (await computes.get(compute.id)).placement
        assert placement is not None
        assert "docker API" in placement.reason
        assert placement.retry_at > now(), "a compute that cannot buy waits before asking again"


def describe_a_compute_refused_by_every_market_and_region() -> None:
    async def it_does_not_call_the_provider_until_the_first_wait_has_passed(refused: Refused) -> None:
        await refused.refused()

        refused.clock.now += REFUSED_FIRST - 0.5
        assert not await refused.attempt()
        assert refused.provider.launches == 1

        refused.clock.now += 0.5
        assert await refused.attempt()

    async def it_doubles_the_wait_after_each_further_refusal(refused: Refused) -> None:
        await refused.refused()

        assert await refused.window() == REFUSED_FIRST
        assert await refused.window() == REFUSED_FIRST * 2
        assert await refused.window() == REFUSED_FIRST * 4

    async def it_stops_doubling_at_the_longest_wait(refused: Refused) -> None:
        await refused.refused()

        windows = [await refused.window() for _ in range(8)]

        assert windows[-2:] == [REFUSED_MAX, REFUSED_MAX]
        assert max(windows) == REFUSED_MAX

    async def it_starts_over_after_a_successful_purchase(refused: Refused) -> None:
        await refused.refused()
        assert await refused.window() == REFUSED_FIRST
        assert await refused.window() == REFUSED_FIRST * 2

        refused.provider.selling = True
        refused.clock.now += REFUSED_FIRST * 4
        assert await refused.attempt()
        refused.provider.selling = False

        assert await refused.attempt()
        assert await refused.window() == REFUSED_FIRST

    async def it_says_on_the_compute_why_no_machine_could_be_bought(refused: Refused) -> None:
        await refused.refused()

        placement = (await ComputeStore(EventStore(), NodeStore()).get(refused.compute_id)).placement
        assert placement is not None
        assert "no capacity" in placement.reason, "the reason names what the provider said, not just that something failed"
        assert placement.retry_at > now(), "a compute waiting to try again says when"

    async def it_stops_saying_so_once_a_machine_is_bought(refused: Refused) -> None:
        await refused.refused()
        refused.provider.selling = True
        refused.clock.now += REFUSED_FIRST
        assert await refused.attempt()

        assert (await ComputeStore(EventStore(), NodeStore()).get(refused.compute_id)).placement is None

    async def it_starts_over_after_release(refused: Refused) -> None:
        await refused.refused()
        assert await refused.window() == REFUSED_FIRST

        await refused.machines.release(refused.compute_id)

        assert await refused.attempt()
        assert await refused.window() == REFUSED_FIRST
