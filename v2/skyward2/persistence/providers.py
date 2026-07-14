import uuid
from datetime import UTC, datetime

from skyward2.application.errors import CapabilityMismatchError, IdempotencyConflictError, NotFoundError
from skyward2.application.provider import Catalog
from skyward2.persistence.tables import OfferRow, ProviderRow
from skyward2.protocol.schemas import Error, Page, Provider, ProviderCreate
from skyward2.providers import adapter_for


class ProviderStore:
    async def create(self, body: ProviderCreate) -> Provider:
        adapter = adapter_for(body.kind)

        if missing := [f for f in adapter.credential_fields if not body.credentials.get(f)]:
            raise CapabilityMismatchError(
                f"{body.kind} requires credentials: {', '.join(adapter.credential_fields)}",
                missing=missing,
            )

        if await ProviderRow.exists().where(ProviderRow.name == body.name):
            raise IdempotencyConflictError(f"provider name already taken: {body.name}")

        row = ProviderRow(
            id=f"prv_{uuid.uuid4().hex[:12]}",
            name=body.name,
            kind=body.kind,
            credentials=dict(body.credentials),
            config=dict(body.config),
            created_at=datetime.now(UTC),
        )
        await row.save().run()
        return await self.get(row.id)

    async def get(self, ref: str) -> Provider:
        row = await ProviderRow.objects().output(load_json=True).where((ProviderRow.id == ref) | (ProviderRow.name == ref)).first()
        if row is None:
            raise NotFoundError(f"no such provider: {ref}")
        return await _to_provider(row)

    async def list(self) -> Page[Provider]:
        rows = await ProviderRow.objects().output(load_json=True).order_by(ProviderRow.created_at)
        return Page(items=tuple([await _to_provider(row) for row in rows]))

    async def delete(self, ref: str) -> None:
        provider = await self.get(ref)
        await OfferRow.delete().where(OfferRow.provider_id == provider.id).run()
        await ProviderRow.delete().where(ProviderRow.id == provider.id).run()

    async def adapter(self, ref: str) -> Catalog:
        """Build the live adapter for a stored provider.

        Credentials are read here and nowhere else. The adapter receives them;
        it never goes looking for an env var of its own.
        """
        row = await ProviderRow.objects().output(load_json=True).where((ProviderRow.id == ref) | (ProviderRow.name == ref)).first()
        if row is None:
            raise NotFoundError(f"no such provider: {ref}")
        return adapter_for(row.kind).create(row.id, row.name, row.credentials, row.config)


async def _to_provider(row: ProviderRow) -> Provider:
    adapter = adapter_for(row.kind)
    count = await OfferRow.count().where(OfferRow.provider_id == row.id)
    return Provider(
        id=row.id,
        name=row.name,
        kind=row.kind,
        config=row.config,
        offers_ttl_seconds=int(adapter.offers_ttl.total_seconds()),
        created_at=row.created_at,
        offers_fetched_at=row.offers_fetched_at,
        offers_count=count,
        last_error=Error(code="unsupported_provider", message=row.last_error, retryable=True) if row.last_error else None,
    )
