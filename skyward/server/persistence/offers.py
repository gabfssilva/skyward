from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import UTC, datetime
from itertools import batched

import msgspec
from piccolo.engine.sqlite import TransactionType

from skyward.server.persistence.providers import ProviderStore
from skyward.server.persistence.tables import OfferRow, ProviderRow
from skyward.shared.accelerators import resolve
from skyward.shared.schemas import BillingUnit, Offer, Page

logger = logging.getLogger(__name__)

BATCH_SIZE = 500
"""Rows per INSERT. SQLite caps a statement at 32766 bound variables, and an
offer binds ~20 of them: Vultr's 3000-row catalog overflows a single insert."""


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
    ) -> Page[Offer]:
        targets = await self._targets(provider, kind)
        if not targets:
            return Page(items=())

        await asyncio.gather(*(self._ensure_fresh(row, force=refresh) for row in targets))

        query = OfferRow.objects().output(load_json=True).where(OfferRow.provider_id.is_in([row.id for row in targets]))
        if accelerator:
            query = query.where(OfferRow.accelerator == resolve(accelerator)[0])
        if min_count:
            query = query.where(OfferRow.accelerator_count >= min_count)
        if min_vram:
            query = query.where(OfferRow.vram >= min_vram)
        if max_price:
            query = query.where(OfferRow.price <= max_price)

        rows = await query.order_by(OfferRow.price)
        return Page(items=tuple(_to_offer(row) for row in rows))

    async def _targets(self, provider: str | None, kind: str | None) -> list[ProviderRow]:
        query = ProviderRow.objects().output(load_json=True)
        if provider:
            query = query.where((ProviderRow.id == provider) | (ProviderRow.name == provider))
        if kind:
            query = query.where(ProviderRow.kind == kind)
        return await query

    async def _ensure_fresh(self, row: ProviderRow, force: bool) -> None:
        async with self._locks[row.id]:
            if not force and not await self._is_stale(row.id):
                return
            try:
                await self._refresh(row)
            except Exception as exc:
                logger.warning("offers refresh failed for %s: %s", row.name, exc)
                await ProviderRow.update({ProviderRow.last_error: str(exc)}).where(ProviderRow.id == row.id).run()

    async def _is_stale(self, provider_id: str) -> bool:
        fresh = await OfferRow.count().where(
            (OfferRow.provider_id == provider_id) & (OfferRow.expires_at > datetime.now(UTC)),
        )
        return fresh == 0

    async def _refresh(self, row: ProviderRow) -> None:
        adapter = await self._providers.adapter(row.id)
        offers = [offer async for offer in adapter.offers()]

        async with OfferRow._meta.db.transaction(transaction_type=TransactionType.immediate):
            await OfferRow.delete().where(OfferRow.provider_id == row.id).run()
            for batch in batched(offers, BATCH_SIZE):
                await OfferRow.insert(*(_to_row(offer) for offer in batch)).run()
            await ProviderRow.update(
                {ProviderRow.offers_fetched_at: datetime.now(UTC), ProviderRow.last_error: None},
            ).where(ProviderRow.id == row.id).run()


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


def _to_offer(row: OfferRow) -> Offer:
    return Offer(
        id=row.offer_id,
        provider_id=row.provider_id,
        provider_name=row.provider_name,
        kind=row.kind,
        instance_type=row.instance_type,
        accelerator=row.accelerator,
        accelerator_count=row.accelerator_count,
        vram=row.vram,
        cpus=row.cpus,
        memory_gb=row.memory_gb,
        disk_gb=row.disk_gb,
        architecture=row.architecture,
        region=row.region,
        spot_price=row.spot_price,
        on_demand_price=row.on_demand_price,
        billing_unit=msgspec.convert(row.billing_unit, BillingUnit),
        available=row.available,
        specific=row.specific,
        fetched_at=row.fetched_at,
        expires_at=row.expires_at,
    )
