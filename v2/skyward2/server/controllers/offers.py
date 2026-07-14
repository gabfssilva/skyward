from litestar import Controller, get
from litestar.params import Parameter

from skyward2.application import ports
from skyward2.protocol.schemas import Offer, Page


class OfferController(Controller):
    path = "/offers"
    tags = ["offers"]

    @get(
        summary="Query instance offers",
        description=(
            "Served from the cache. Any provider whose offers have expired is refreshed first — the TTL belongs to the "
            "provider, because a marketplace bundle is gone in minutes while a fixed instance type is not.\n\n"
            "A refresh that fails does not empty the catalog: the stale rows are still served and the failure shows up "
            "as `last_error` on the provider. A provider being down degrades the answer; it does not erase it."
        ),
    )
    async def list(
        self,
        offers: ports.Offers,
        provider: str | None = None,
        kind: str | None = None,
        accelerator: str | None = None,
        min_count: int | None = None,
        max_price: float | None = None,
        refresh: bool = Parameter(default=False, description="Force a refetch even if the cache is still within its TTL."),
    ) -> Page[Offer]:
        return await offers.list(provider, kind, accelerator, min_count, max_price, refresh)
