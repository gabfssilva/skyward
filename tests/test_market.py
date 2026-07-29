"""Which offers a spec will admit — the filters that read the machine, not the list price.

``disk_gb`` and ``architecture`` read the offer, because both are facts about the
machine. ``max_hourly_cost`` reads the *buy*, because the cap is on what the machine is
actually billed at: an offer whose spot price fits and whose on-demand price does not is
still worth having on spot.

The two machine filters agree on the unreported offer, and that agreement is the point:
one that does not say what it is cannot be shown to satisfy a floor or an instruction
set, so it is dropped rather than gambled on.
"""

from datetime import UTC, datetime

from skyward.server.application.market import _candidates
from skyward.shared.schemas import ComputeSpec, NodeBounds, Offer, Page, ProviderRef, Spec

NOW = datetime.now(UTC)


def offer(name: str, disk: float | None, spot: float | None, on_demand: float | None, arch: str | None = None) -> Offer:
    return Offer(
        architecture=arch,
        id=name,
        provider_id="prv",
        provider_name="test",
        kind="fake",
        instance_type=name,
        accelerator="h100",
        accelerator_count=1,
        cpus=8,
        memory_gb=64,
        disk_gb=disk,
        spot_price=spot,
        on_demand_price=on_demand,
        fetched_at=NOW,
        expires_at=NOW,
    )


class FakeOffers:
    """Only the one method :func:`_candidates` reaches for."""

    def __init__(self, *items: Offer) -> None:
        self._items = items

    async def list(self, **_: object) -> Page[Offer]:
        return Page(items=self._items)


def spec(**wanted: object) -> ComputeSpec:
    return ComputeSpec(specs=(Spec(provider=ProviderRef(kind="fake"), **wanted),), nodes=NodeBounds(desired=1))  # type: ignore[arg-type]


async def test_a_disk_floor_drops_the_machines_below_it():
    offers = FakeOffers(offer("small", 100, 1.0, 2.0), offer("big", 500, 1.0, 2.0))

    picked = await _candidates(spec(disk_gb=200), offers)  # type: ignore[arg-type]

    assert {buy.offer.id for buy in picked} == {"big"}


async def test_an_offer_that_does_not_say_its_disk_cannot_satisfy_a_floor():
    offers = FakeOffers(offer("unknown", None, 1.0, 2.0))

    assert await _candidates(spec(disk_gb=10), offers) == []  # type: ignore[arg-type]


async def test_an_architecture_admits_only_the_machines_that_run_it():
    offers = FakeOffers(offer("intel", 100, 1.0, 2.0, arch="x86_64"), offer("graviton", 100, 1.0, 2.0, arch="arm64"))

    picked = await _candidates(spec(architecture="arm64"), offers)  # type: ignore[arg-type]

    assert {buy.offer.id for buy in picked} == {"graviton"}


async def test_an_offer_that_does_not_say_its_architecture_cannot_satisfy_a_request_for_one():
    """The one behaviour a refactor would get backwards, and the one that breaks a node.

    ``None`` is not a wildcard on the offer's side. An offer whose provider never
    reported an architecture is excluded from a spec that named one, exactly as an
    offer that reported the wrong one is — the alternative is shipping x86 wheels to
    an arm machine and finding out on the first import.
    """
    offers = FakeOffers(offer("unknown", 100, 1.0, 2.0), offer("intel", 100, 1.0, 2.0, arch="x86_64"))

    assert await _candidates(spec(architecture="arm64"), offers) == []  # type: ignore[arg-type]


async def test_a_spec_that_names_no_architecture_admits_the_offers_that_did_not_say():
    offers = FakeOffers(offer("unknown", 100, 1.0, 2.0), offer("graviton", 100, 1.0, 2.0, arch="arm64"))

    picked = await _candidates(spec(), offers)  # type: ignore[arg-type]

    assert {buy.offer.id for buy in picked} == {"unknown", "graviton"}


async def test_a_cost_cap_drops_the_buys_above_it_and_keeps_the_ones_below():
    offers = FakeOffers(offer("both", 100, 1.0, 9.0))

    picked = await _candidates(spec(max_hourly_cost=5.0), offers)  # type: ignore[arg-type]

    assert [(buy.market, buy.price) for buy in picked] == [("spot", 1.0)]


async def test_a_spec_that_asks_for_nothing_admits_every_way_to_buy():
    offers = FakeOffers(offer("a", 100, 1.0, 2.0))

    assert len(await _candidates(spec(), offers)) == 2  # type: ignore[arg-type]
