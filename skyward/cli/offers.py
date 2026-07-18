"""sky offers — what a GPU costs, and where.

Every command here is one GET against ``/v1/offers``. The daemon owns the
catalog, the per-provider TTL, and the refresh that a stale provider triggers,
so the only thing left on this side is turning flags into query parameters and
rows into columns.

``fetch`` is not a separate route: forcing a refetch is ``refresh=true`` on the
same query, which is why a fetch can also filter.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from math import inf
from pathlib import Path

from skyward.cli import offers_app
from skyward.cli._client import call
from skyward.cli._output import EMPTY, Output, render
from skyward.protocol.schemas import Offer, Page

LIST_COLUMNS = ("PROVIDER", "KIND", "INSTANCE", "ACCELERATOR", "VRAM", "CPUS", "MEMORY", "REGION", "SPOT", "ON-DEMAND", "UNIT")
SUMMARY_COLUMNS = ("ACCELERATOR", "PROVIDER", "OFFERS", "CHEAPEST", "AVERAGE", "DEAREST")


@offers_app.command(name="list")
def list_offers(
    *,
    provider: str | None = None,
    accelerator: str | None = None,
    min_count: int | None = None,
    min_vram: float | None = None,
    max_price: float | None = None,
    limit: int = 20,
    refresh: bool = False,
    output: Output = "table",
    url: str | None = None,
    database: Path | None = None,
) -> None:
    """List offers, cheapest first.

    Parameters
    ----------
    provider
        Provider id or name.
    accelerator
        Accelerator name, in any spelling the provider uses.
    min_count
        Least acceptable number of accelerators per instance.
    min_vram
        Least acceptable VRAM per accelerator, in GB.
    max_price
        Most the offer may cost, in its own billing unit.
    limit
        Rows to print. Zero prints everything.
    refresh
        Refetch before answering, even if the cache is still fresh.
    output
        ``table`` for a person, ``json`` for a program.
    url
        The daemon to ask. Defaults to ``SKYWARD_URL``, else an embedded one.
    database
        Where the embedded daemon keeps its state.
    """
    page = _query(provider, accelerator, min_count, min_vram, max_price, refresh, url, database)
    offers = page.items if limit <= 0 else page.items[:limit]
    render(LIST_COLUMNS, [_row(offer) for offer in offers], output=output)


@offers_app.command(name="fetch")
def fetch_offers(
    *,
    provider: str | None = None,
    output: Output = "table",
    url: str | None = None,
    database: Path | None = None,
) -> None:
    """Refetch the catalog and report how many offers each provider returned.

    A provider that fails to answer keeps its stale rows and reports the failure
    on itself, so a count here is what the catalog holds, not what the fetch won.

    Parameters
    ----------
    provider
        Provider id or name. Defaults to every configured provider.
    output
        ``table`` for a person, ``json`` for a program.
    url
        The daemon to ask. Defaults to ``SKYWARD_URL``, else an embedded one.
    database
        Where the embedded daemon keeps its state.
    """
    page = _query(provider, None, None, None, None, True, url, database)
    counts = Counter(offer.provider_name for offer in page.items)
    render(("PROVIDER", "OFFERS"), sorted(counts.items()), output=output)


@offers_app.command(name="summary")
def summary_offers(
    *,
    provider: str | None = None,
    accelerator: str | None = None,
    min_count: int | None = None,
    min_vram: float | None = None,
    max_price: float | None = None,
    refresh: bool = False,
    output: Output = "table",
    url: str | None = None,
    database: Path | None = None,
) -> None:
    """Aggregate the catalog by accelerator and provider.

    Parameters
    ----------
    provider
        Provider id or name.
    accelerator
        Accelerator name, in any spelling the provider uses.
    min_count
        Least acceptable number of accelerators per instance.
    min_vram
        Least acceptable VRAM per accelerator, in GB.
    max_price
        Most the offer may cost, in its own billing unit.
    refresh
        Refetch before answering, even if the cache is still fresh.
    output
        ``table`` for a person, ``json`` for a program.
    url
        The daemon to ask. Defaults to ``SKYWARD_URL``, else an embedded one.
    database
        Where the embedded daemon keeps its state.
    """
    page = _query(provider, accelerator, min_count, min_vram, max_price, refresh, url, database)

    groups: defaultdict[tuple[str, str], list[Offer]] = defaultdict(list)
    for offer in page.items:
        groups[(offer.accelerator or EMPTY, offer.provider_name)].append(offer)

    ordered = sorted(groups.items(), key=lambda group: (group[0][0], _cheapest(group[1])))
    render(SUMMARY_COLUMNS, [_summary(key, offers) for key, offers in ordered], output=output)


def _query(
    provider: str | None,
    accelerator: str | None,
    min_count: int | None,
    min_vram: float | None,
    max_price: float | None,
    refresh: bool,
    url: str | None,
    database: Path | None,
) -> Page[Offer]:
    return call(
        lambda client: client.call(
            "GET",
            "/v1/offers",
            Page[Offer],
            provider=provider,
            accelerator=accelerator,
            min_count=min_count,
            min_vram=min_vram,
            max_price=max_price,
            refresh=refresh or None,
        ),
        url=url,
        database=database,
    )


def _row(offer: Offer) -> tuple[object, ...]:
    return (
        offer.provider_name,
        offer.kind,
        offer.instance_type,
        _accelerator(offer),
        _amount(offer.vram, "GB"),
        offer.cpus,
        _amount(offer.memory_gb, "GB"),
        offer.region,
        _price(offer.spot_price),
        _price(offer.on_demand_price),
        offer.billing_unit,
    )


def _summary(key: tuple[str, str], offers: Sequence[Offer]) -> tuple[object, ...]:
    accelerator, provider = key
    prices = [offer.price for offer in offers if offer.price is not None]
    average = sum(prices) / len(prices) if prices else None
    return (
        accelerator,
        provider,
        len(offers),
        _price(min(prices, default=None)),
        _price(average),
        _price(max(prices, default=None)),
    )


def _cheapest(offers: Sequence[Offer]) -> float:
    return min((offer.price for offer in offers if offer.price is not None), default=inf)


def _accelerator(offer: Offer) -> str | None:
    match offer.accelerator:
        case None:
            return None
        case name if offer.accelerator_count == 1:
            return name
        case name:
            return f"{offer.accelerator_count}x {name}"


def _amount(value: float | None, unit: str) -> str | None:
    return None if value is None else f"{value:g}{unit}"


def _price(value: float | None) -> str | None:
    return None if value is None else f"{value:.4f}".rstrip("0").rstrip(".")


__all__ = ["fetch_offers", "list_offers", "summary_offers"]
