"""Which market a pool is bought on.

The offer and the market are one decision, so the policy is one function. What is
tested here is that asking for spot never quietly buys on demand, that
``spot_if_available`` falls back rather than failing, and that ``cheapest`` is
allowed to pick either.
"""

from datetime import UTC, datetime

import pytest

from skyward.server.application.market import _buys, _cheapest
from skyward.shared.schemas import Allocation, Market, Offer

NOW = datetime.now(UTC)


def offer(name: str, spot: float | None, on_demand: float | None) -> Offer:
    return Offer(
        id=name,
        provider_id="prv_1",
        provider_name="test",
        kind="fake",
        instance_type=name,
        accelerator="h100",
        accelerator_count=1,
        cpus=8,
        memory_gb=64,
        spot_price=spot,
        on_demand_price=on_demand,
        fetched_at=NOW,
        expires_at=NOW,
    )


BOTH = offer("both", spot=1.0, on_demand=4.0)
SPOT_ONLY = offer("spot-only", spot=3.0, on_demand=None)
ON_DEMAND_ONLY = offer("on-demand-only", spot=None, on_demand=2.0)


@pytest.mark.parametrize(
    ("allocation", "expected"),
    [
        ("spot", [(BOTH, "spot"), (SPOT_ONLY, "spot")]),
        ("on_demand", [(BOTH, "on_demand"), (ON_DEMAND_ONLY, "on_demand")]),
        ("cheapest", [(BOTH, "spot"), (BOTH, "on_demand"), (SPOT_ONLY, "spot"), (ON_DEMAND_ONLY, "on_demand")]),
    ],
)
def test_an_allocation_says_how_each_offer_may_be_bought(allocation: Allocation, expected: list[tuple[Offer, Market]]):
    offers = (BOTH, SPOT_ONLY, ON_DEMAND_ONLY)
    bought = [(buy.offer, buy.market) for wanted in offers for buy in _buys(wanted, allocation)]

    assert bought == expected


def test_an_offer_with_no_spot_price_is_not_a_spot_offer():
    assert _buys(ON_DEMAND_ONLY, "spot") == []


def test_cheapest_takes_the_price_whichever_market_it_is_on():
    buys = [buy for wanted in (BOTH, ON_DEMAND_ONLY) for buy in _buys(wanted, "cheapest")]

    assert _cheapest(buys, "cheapest") == (BOTH, "spot")


def test_spot_if_available_prefers_spot_even_when_on_demand_is_cheaper():
    buys = [buy for wanted in (SPOT_ONLY, ON_DEMAND_ONLY) for buy in _buys(wanted, "spot_if_available")]

    assert _cheapest(buys, "spot_if_available") == (SPOT_ONLY, "spot"), "3.00 spot over 2.00 on demand: it was asked for"


def test_spot_if_available_falls_back_when_nothing_is_on_the_spot_market():
    buys = _buys(ON_DEMAND_ONLY, "spot_if_available")

    assert _cheapest(buys, "spot_if_available") == (ON_DEMAND_ONLY, "on_demand")
