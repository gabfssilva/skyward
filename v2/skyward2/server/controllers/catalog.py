from __future__ import annotations

from litestar import Controller, get

from skyward2.application import ports
from skyward2.protocol.schemas import Offer, Provider


class CatalogController(Controller):
    tags = ["catalog"]

    @get(
        "/providers",
        summary="List providers and their capabilities",
        description=(
            "Read-only. This is what capability negotiation runs against before any external effect: a provider without "
            "tags, tag lookup and a renewable lease cannot recover a create whose commit was lost, so it does not "
            "support daemon mode and is refused at creation — not halfway through provisioning."
        ),
    )
    async def providers(self, catalog: ports.Catalog) -> tuple[Provider, ...]:
        return await catalog.providers()

    @get("/offers", summary="Query instance offers")
    async def offers(
        self,
        catalog: ports.Catalog,
        provider: str | None = None,
        accelerator: str | None = None,
        min_count: int | None = None,
    ) -> tuple[Offer, ...]:
        return await catalog.offers(provider, accelerator, min_count)
