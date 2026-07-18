"""Which machine to buy, and at which price.

A decision, taken once per compute, out of a catalogue. It reads nothing and
writes nothing — everything it needs arrives as an argument, which is why it is
here and not inside the thing that provisions.
"""

from __future__ import annotations

from typing import NamedTuple

from skyward.application.errors import CapabilityMismatchError
from skyward.persistence.offers import OfferCache
from skyward.protocol.schemas import Allocation, ComputeSpec, Market, Offer


class Buy(NamedTuple):
    """One way to have one offer: the machine, the market, and what that costs."""

    offer: Offer
    market: Market
    price: float


async def pick(spec: ComputeSpec, offers: OfferCache) -> tuple[Offer, Market]:
    """The hardware, out of everything on offer that would do — and at which price.

    Each ``Spec`` is an alternative, not a requirement — the point of listing
    several is that the second one is what you get when the first is sold out.

    The market comes out of the same decision as the offer, because it is not
    separable from it: an offer sold only on the spot market is not a cheaper way
    to buy the same machine, it is a different machine to be billed for.
    """
    candidates = await _candidates(spec, offers)
    if not candidates:
        raise CapabilityMismatchError(f"nothing on offer satisfies the spec on the {spec.allocation} market")

    return _cheapest(candidates, spec.allocation)


async def rank(spec: ComputeSpec, offers: OfferCache) -> tuple[Offer, ...]:
    """Every offer that fits, cheapest first — the order to try regions when one refuses.

    The same catalogue :func:`pick` reads, deduplicated to one entry per offer and
    ordered by price, so a placement that has exhausted the markets of its bound
    region can walk to the next cheapest one that might still sell. The offer
    already bound is in here too; the caller skips it rather than buy it twice.
    """
    ordered: dict[str, Offer] = {}
    for buy in sorted(await _candidates(spec, offers), key=lambda buy: buy.price):
        ordered.setdefault(buy.offer.id, buy.offer)
    return tuple(ordered.values())


async def _candidates(spec: ComputeSpec, offers: OfferCache) -> list[Buy]:
    """Every way to buy every offer that fits the spec, across all its alternatives."""
    candidates: list[Buy] = []

    for wanted in spec.specs:
        page = await offers.list(
            provider=None,
            kind=wanted.provider.kind,
            accelerator=wanted.accelerator,
            min_count=wanted.accelerator_count if wanted.accelerator else None,
            min_vram=None,
            max_price=None,
            refresh=False,
        )
        fitting = [
            offer for offer in page.items
            if (wanted.cpus is None or offer.cpus >= wanted.cpus)
            and (wanted.memory_gb is None or offer.memory_gb >= wanted.memory_gb)
            and (wanted.region is None or offer.region == wanted.region)
            and (wanted.disk_gb is None or (offer.disk_gb is not None and offer.disk_gb >= wanted.disk_gb))
        ]
        buys = [
            buy
            for offer in fitting
            for buy in _buys(offer, spec.allocation)
            if wanted.max_hourly_cost is None or buy.price <= wanted.max_hourly_cost
        ]
        if buys and spec.selection == "first":
            return buys
        candidates.extend(buys)

    return candidates


def order(offer: Offer, allocation: Allocation) -> tuple[Market, ...]:
    """The markets to try for this offer, in the order to try them.

    A liquid decision, not a frozen one: ``spot_if_available`` leads with spot and
    keeps on-demand as the fallback for the node whose spot launch is refused,
    ``cheapest`` leads with whichever is cheaper. The single-market allocations
    yield their one market. Empty only if the offer carries no price the allocation
    can buy — the same emptiness :func:`pick` raises on.
    """
    buys = _buys(offer, allocation)
    if allocation == "cheapest":
        buys = sorted(buys, key=lambda buy: buy.price)
    return tuple(buy.market for buy in buys)


def _buys(offer: Offer, allocation: Allocation) -> list[Buy]:
    """What the allocation allows this offer to be bought as, if anything.

    An offer with no spot price is not a spot offer, and asking for spot excludes
    it rather than silently buying it on demand — the price the pool was chosen on
    has to be the price it is billed at.
    """
    spot = Buy(offer, "spot", offer.spot_price) if offer.spot_price is not None else None
    on_demand = Buy(offer, "on_demand", offer.on_demand_price) if offer.on_demand_price is not None else None

    match allocation:
        case "spot":
            return [buy for buy in (spot,) if buy]
        case "on_demand":
            return [buy for buy in (on_demand,) if buy]
        case "spot_if_available" | "cheapest":
            return [buy for buy in (spot, on_demand) if buy]


def _cheapest(buys: list[Buy], allocation: Allocation) -> tuple[Offer, Market]:
    """``spot_if_available`` prefers the spot market; every other allocation prefers the price."""
    preferred = [buy for buy in buys if buy.market == "spot"] if allocation == "spot_if_available" else []
    buy = min(preferred or buys, key=lambda buy: buy.price)
    return buy.offer, buy.market
