"""Which offers a spec will admit — the two filters that read the machine, not the list price.

``disk_gb`` reads the offer, because disk is a fact about the machine. ``max_hourly_cost``
reads the *buy*, because the cap is on what the machine is actually billed at: an offer
whose spot price fits and whose on-demand price does not is still worth having on spot.
"""

from datetime import UTC, datetime

from skyward.application.market import _candidates
from skyward.protocol.schemas import ComputeSpec, NodeBounds, Offer, Page, ProviderRef, Spec

NOW = datetime.now(UTC)


def offer(name: str, disk: float | None, spot: float | None, on_demand: float | None) -> Offer:
    return Offer(
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


async def test_a_cost_cap_drops_the_buys_above_it_and_keeps_the_ones_below():
    offers = FakeOffers(offer("both", 100, 1.0, 9.0))

    picked = await _candidates(spec(max_hourly_cost=5.0), offers)  # type: ignore[arg-type]

    assert [(buy.market, buy.price) for buy in picked] == [("spot", 1.0)]


async def test_a_spec_that_asks_for_nothing_admits_every_way_to_buy():
    offers = FakeOffers(offer("a", 100, 1.0, 2.0))

    assert len(await _candidates(spec(), offers)) == 2  # type: ignore[arg-type]
