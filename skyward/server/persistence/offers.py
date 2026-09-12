from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from itertools import batched
from typing import Any

import msgspec
from piccolo.columns import Column
from piccolo.custom_types import Combinable
from piccolo.query.mixins import OrderByRaw

from skyward.providers.registry import adapter_for
from skyward.server.persistence.db import transaction
from skyward.server.persistence.providers import ProviderStore
from skyward.server.persistence.tables import OfferRow, ProviderRow
from skyward.shared.accelerators import resolve
from skyward.shared.observability import logger
from skyward.shared.schemas import BillingUnit, Offer, OfferSort, Page

logger = logger.bind(component="offers")

BATCH_SIZE = 500
"""Rows per INSERT. SQLite caps a statement at 32766 bound variables, and an
offer binds ~20 of them: Vultr's 3000-row catalog overflows a single insert."""

RETRY_SECONDS = 60.0
"""A catalog that failed is asked for again after this, or its TTL if shorter."""

ORDERS: dict[OfferSort, tuple[Column | OrderByRaw, bool]] = {
    "price": (OrderByRaw("(price IS NULL), price / max(accelerator_count, 1)"), True),
    "vram": (OfferRow.vram, False),
    "available": (OfferRow.available, False),
}
"""How each order is written, and which way it runs.

Price is per accelerator — the only comparison that holds between offers selling
different numbers of them — and an offer with no price at all is ordered last
rather than first, where dividing nothing would put it.
"""

_specific = msgspec.json.Decoder(dict[str, Any])


class OfferCache:
    """Offers, cached per provider, with the provider deciding staleness.

    The TTL is not a cache-tuning knob: a Vast.ai bundle really is gone in
    minutes, and an AWS instance type really is not. Letting the cache pick one
    number for everyone would make it either wrong or useless.
    """

    def __init__(self, providers: ProviderStore) -> None:
        self._providers = providers
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def list(
        self,
        provider: str | None,
        kind: str | None,
        accelerator: str | None,
        min_count: int | None,
        min_vram: float | None,
        max_price: float | None,
        refresh: bool,
        *,
        spot: bool | None = None,
        sort: OfferSort = "price",
        limit: int | None = None,
    ) -> Page[Offer]:
        targets = await self._targets(provider, kind)
        if not targets:
            return Page(items=(), total=0)

        await asyncio.gather(*(self._ensure_fresh(row, force=refresh) for row in targets))

        narrowed: list[Combinable] = [OfferRow.provider_id.is_in([row.id for row in targets])]
        if accelerator:
            narrowed.append(OfferRow.accelerator == resolve(accelerator)[0])
        if min_count:
            narrowed.append(OfferRow.accelerator_count >= min_count)
        if min_vram:
            narrowed.append(OfferRow.vram >= min_vram)
        if max_price:
            narrowed.append(OfferRow.price <= max_price)
        if spot:
            narrowed.append(OfferRow.spot_price.is_not_null())
        elif spot is False:
            narrowed.append(OfferRow.spot_price.is_null())

        column, ascending = ORDERS[sort]
        ordered = OfferRow.select(*OfferRow.all_columns()).where(*narrowed).order_by(column, ascending=ascending)
        rows = await (ordered.limit(limit) if limit else ordered)
        return Page(items=tuple(_to_offer(row) for row in rows), total=await OfferRow.count().where(*narrowed))

    async def _targets(self, provider: str | None, kind: str | None) -> list[ProviderRow]:
        query = ProviderRow.objects().output(load_json=True)
        if provider:
            query = query.where((ProviderRow.id == provider) | (ProviderRow.name == provider))
        if kind:
            query = query.where(ProviderRow.kind == kind)
        return await query

    async def _ensure_fresh(self, row: ProviderRow, force: bool) -> None:
        async with self._locks[row.id]:
            current = await ProviderRow.select(ProviderRow.offers_attempted_at, ProviderRow.last_error).where(ProviderRow.id == row.id).first()
            failed = current is not None and current["last_error"] is not None
            if not force and current is not None and not _due(current["offers_attempted_at"], failed, adapter_for(row.kind).offers_ttl):
                return
            try:
                await self._refresh(row)
            except Exception as exc:
                logger.bind(provider=row.name).warning("offers refresh failed: {}", exc)
                await ProviderRow.update(
                    {ProviderRow.last_error: str(exc), ProviderRow.offers_attempted_at: datetime.now(UTC)},
                ).where(ProviderRow.id == row.id).run()

    async def _refresh(self, row: ProviderRow) -> None:
        log = logger.bind(provider=row.name)
        log.debug("catalogue is stale, fetching")
        adapter = await self._providers.adapter(row.id)
        offers = [offer async for offer in adapter.offers()]
        log.info("{} offers", len(offers))
        rows = [_to_row(offer) for offer in offers]

        async with transaction():
            await OfferRow.delete().where(OfferRow.provider_id == row.id).run()
            for batch in batched(rows, BATCH_SIZE):
                await OfferRow.insert(*batch).run()
            now = datetime.now(UTC)
            await ProviderRow.update(
                {ProviderRow.offers_fetched_at: now, ProviderRow.offers_attempted_at: now, ProviderRow.last_error: None},
            ).where(ProviderRow.id == row.id).run()


def _due(attempted: datetime | None, failed: bool, ttl: timedelta) -> bool:
    if attempted is None:
        return True
    wait = min(ttl, timedelta(seconds=RETRY_SECONDS)) if failed else ttl
    return datetime.now(UTC) - attempted >= wait


def _to_row(offer: Offer) -> OfferRow:
    return OfferRow(
        id=f"{offer.provider_id}:{offer.id}",
        offer_id=offer.id,
        provider_id=offer.provider_id,
        provider_name=offer.provider_name,
        kind=offer.kind,
        instance_type=offer.instance_type,
        accelerator=offer.accelerator,
        accelerator_count=offer.accelerator_count,
        vram=offer.vram,
        cpus=offer.cpus,
        memory_gb=offer.memory_gb,
        disk_gb=offer.disk_gb,
        architecture=offer.architecture,
        region=offer.region,
        spot_price=offer.spot_price,
        on_demand_price=offer.on_demand_price,
        price=offer.price,
        billing_unit=offer.billing_unit,
        available=offer.available,
        specific=offer.specific,
        fetched_at=offer.fetched_at,
        expires_at=offer.expires_at,
    )


def _to_offer(row: dict[str, Any]) -> Offer:
    return Offer(
        id=row["offer_id"],
        provider_id=row["provider_id"],
        provider_name=row["provider_name"],
        kind=row["kind"],
        instance_type=row["instance_type"],
        accelerator=row["accelerator"],
        accelerator_count=row["accelerator_count"],
        vram=row["vram"],
        cpus=row["cpus"],
        memory_gb=row["memory_gb"],
        disk_gb=row["disk_gb"],
        architecture=row["architecture"],
        region=row["region"],
        spot_price=row["spot_price"],
        on_demand_price=row["on_demand_price"],
        billing_unit=msgspec.convert(row["billing_unit"], BillingUnit),
        available=row["available"],
        specific=_specific.decode(row["specific"]),
        fetched_at=row["fetched_at"],
        expires_at=row["expires_at"],
    )
