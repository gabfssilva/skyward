"""Which machine to buy, and at which price.

A decision, taken once per compute, out of a catalogue. It reads nothing and
writes nothing — everything it needs arrives as an argument, which is why it is
here and not inside the thing that provisions.
"""

from __future__ import annotations

from typing import NamedTuple

from skyward.server.persistence.offers import OfferCache
from skyward.shared.errors import CapabilityMismatchError
from skyward.shared.observability import logger
from skyward.shared.schemas import Allocation, ComputeSpec, Market, Offer, Spec

logger = logger.bind(component="market")


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
        raise CapabilityMismatchError(f"nothing on offer satisfies the spec on the {spec.allocation} market{_because(spec)}")

    buy = _cheapest(candidates, spec.allocation)
    logger.bind(provider=buy.offer.provider_name).info(
        "picked {} in {} on the {} market at ${:.3f}/h, out of {} ways to buy",
        buy.offer.instance_type,
        buy.offer.region or "the account's default region",
        buy.market,
        buy.price,
        len(candidates),
    )
    return buy.offer, buy.market


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
    """Every way to buy every offer that fits the spec, across all its alternatives.

    The accelerator count is the one requirement matched exactly rather than as a
    minimum. A provider that sells the same GPU in every multiple — RunPod lists
    a 1x through an 8x of each — turns "more is also fine" into a ladder the
    fallback walks: the 1x refused for want of capacity, the next cheapest offer
    is the same GPU three at a time, at three times the price. Asking for one
    means one.
    """
    candidates: list[Buy] = []

    for wanted in spec.specs:
        page = await offers.list(
            provider=wanted.provider.name,
            kind=wanted.provider.kind,
            accelerator=wanted.accelerator,
            min_count=wanted.accelerator_count if wanted.accelerator else None,
            min_vram=None,
            max_price=None,
            refresh=False,
        )
        fitting = [
            offer for offer in page.items
            if _accelerators(offer, wanted)
            and (wanted.cpus is None or offer.cpus >= wanted.cpus)
            and (wanted.memory_gb is None or offer.memory_gb >= wanted.memory_gb)
            and (wanted.region is None or offer.region == wanted.region)
            and (wanted.disk_gb is None or (offer.disk_gb is not None and offer.disk_gb >= wanted.disk_gb))
            and (wanted.architecture is None or offer.architecture == wanted.architecture)
        ]
        buys = [
            buy
            for offer in fitting
            for buy in _buys(offer, spec.allocation)
            if wanted.max_hourly_cost is None or buy.price <= wanted.max_hourly_cost
        ]
        logger.debug(
            "{}: {} offers on the shelf, {} fit the spec, {} ways to buy them",
            wanted.provider.kind,
            len(page.items),
            len(fitting),
            len(buys),
        )
        if buys and spec.selection == "first":
            return buys
        candidates.extend(buys)

    return candidates


def _because(spec: ComputeSpec) -> str:
    """The suspect worth naming when a spec matches nothing: it asked for no accelerator.

    An account selling GPUs and nothing else has no machine without one, and a spec
    that names no accelerator asks for exactly that — where it used to be sold the
    cheapest GPU instead. The floors may equally be what nothing cleared, so this
    points rather than concludes; it is here because it is the one cause a reader
    is unlikely to think of on their own.
    """
    if any(wanted.accelerator for wanted in spec.specs):
        return ""
    return " — it asks for a machine with no accelerator"


def _accelerators(offer: Offer, wanted: Spec) -> bool:
    """Whether the offer carries the accelerators the spec asked for, including none of them.

    A spec that names no accelerator is asking for a machine without one, and the
    count it carries is not consulted: there is no such thing as one accelerator of
    no particular model. It used to mean "anything that clears the other floors",
    which reads the same on a provider whose GPUs cost more than its CPUs and quite
    differently on one that bundles the vCPUs and the RAM into the GPU's price —
    there the cheapest machine clearing a CPU floor is a GPU nobody asked for, on a
    smaller pool of machines, and the pool comes up holding one.
    """
    if wanted.accelerator is None:
        return offer.accelerator is None
    return offer.accelerator_count == wanted.accelerator_count


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


def _cheapest(buys: list[Buy], allocation: Allocation) -> Buy:
    """``spot_if_available`` prefers the spot market; every other allocation prefers the price."""
    preferred = [buy for buy in buys if buy.market == "spot"] if allocation == "spot_if_available" else []
    return min(preferred or buys, key=lambda buy: buy.price)
