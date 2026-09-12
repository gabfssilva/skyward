"""How often the daemon asks a provider for its catalog.

Every offers read goes through the cache, and a provider is a network call that
can be slow, failing or rate-limited. A catalog is asked for again when its TTL
says so, when it failed and a retry is due, or when the account behind it changed
— and never merely because somebody read the offers.
"""

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar, Self

import pytest

from skyward.providers.registry import REGISTRY
from skyward.server.persistence.db import connect
from skyward.server.persistence.offers import RETRY_SECONDS, OfferCache
from skyward.server.persistence.providers import ProviderStore
from skyward.server.persistence.tables import ProviderRow
from skyward.shared.schemas import Offer, OfferSort, Page, Provider, ProviderCreate

pytestmark = pytest.mark.local


@dataclass
class Catalog:
    """What the stub provider answers, and how many times it was asked."""

    offers: tuple[str, ...] = ("stub-a",)
    spot: frozenset[str] = frozenset()
    failure: str | None = None
    ttl: timedelta = timedelta(minutes=10)
    asked: int = 0


@pytest.fixture
def catalog(monkeypatch: pytest.MonkeyPatch) -> Catalog:
    state = Catalog()

    class StubProvider:
        kind: ClassVar[str] = "stub"
        credential_fields: ClassVar[tuple[str, ...]] = ()
        offers_ttl: ClassVar[timedelta] = state.ttl

        def __init__(self, provider_id: str, name: str) -> None:
            self._id = provider_id
            self._name = name

        @classmethod
        def create(cls, provider_id: str, name: str, credentials: Mapping[str, str], config: Mapping[str, Any]) -> Self:
            return cls(provider_id, name)

        async def offers(self) -> AsyncIterator[Offer]:
            state.asked += 1
            if state.failure is not None:
                raise RuntimeError(state.failure)
            now = datetime.now(UTC)
            for index, offer in enumerate(state.offers, start=1):
                yield Offer(
                    id=offer,
                    provider_id=self._id,
                    provider_name=self._name,
                    kind=self.kind,
                    billing_unit="second",
                    instance_type=offer,
                    accelerator=None,
                    accelerator_count=index,
                    cpus=2,
                    memory_gb=4.0,
                    region="nowhere",
                    spot_price=0.05 if offer in state.spot else None,
                    on_demand_price=0.1,
                    available=len(state.offers) - index + 1,
                    fetched_at=now,
                    expires_at=now + StubProvider.offers_ttl,
                    specific={},
                )

    monkeypatch.setitem(REGISTRY, "stub", StubProvider)
    return state


async def registered(database: Path) -> tuple[ProviderStore, OfferCache, Provider]:
    await connect(database)
    providers = ProviderStore()
    provider = await providers.create(ProviderCreate(name="stubby", kind="stub", credentials={}, config={}))
    return providers, OfferCache(providers), provider


async def read(cache: OfferCache, *, spot: bool | None = None, sort: OfferSort = "price", limit: int | None = None) -> Page[Offer]:
    return await cache.list(
        provider="stubby",
        kind=None,
        accelerator=None,
        min_count=None,
        min_vram=None,
        max_price=None,
        refresh=False,
        spot=spot,
        sort=sort,
        limit=limit,
    )


async def attempted_ago(provider: Provider, elapsed: timedelta) -> None:
    await ProviderRow.update({ProviderRow.offers_attempted_at: datetime.now(UTC) - elapsed}).where(ProviderRow.id == provider.id).run()


def describe_a_catalog_that_failed() -> None:
    async def it_is_not_asked_again_on_every_read(tmp_path: Path, catalog: Catalog) -> None:
        providers, cache, provider = await registered(tmp_path / "skyward.sqlite")
        catalog.failure = "rate limited"

        await read(cache)
        await read(cache)
        await read(cache)

        assert catalog.asked == 1, "a failing provider is a retry per window, not one per read"
        last_error = (await providers.get(provider.id)).last_error
        assert last_error is not None and "rate limited" in last_error.message, "the reason stays visible while it waits"

    async def it_is_asked_again_once_the_retry_window_passes(tmp_path: Path, catalog: Catalog) -> None:
        providers, cache, provider = await registered(tmp_path / "skyward.sqlite")
        catalog.failure = "rate limited"
        await read(cache)

        await attempted_ago(provider, timedelta(seconds=RETRY_SECONDS - 5))
        await read(cache)
        assert catalog.asked == 1, "the window is not over yet"

        catalog.failure = None
        await attempted_ago(provider, timedelta(seconds=RETRY_SECONDS + 1))
        offers = await read(cache)

        assert catalog.asked == 2
        assert [offer.id for offer in offers.items] == ["stub-a"]
        assert (await providers.get(provider.id)).last_error is None, "a catalog that came back clears the failure"

    async def it_waits_only_its_ttl_when_that_is_shorter_than_the_retry(tmp_path: Path, catalog: Catalog, monkeypatch: pytest.MonkeyPatch) -> None:
        _, cache, provider = await registered(tmp_path / "skyward.sqlite")
        monkeypatch.setattr(REGISTRY["stub"], "offers_ttl", timedelta(seconds=10))
        catalog.failure = "rate limited"
        await read(cache)

        await attempted_ago(provider, timedelta(seconds=11))
        await read(cache)

        assert catalog.asked == 2, "a catalog true for ten seconds is not held back for a minute"


def describe_an_empty_catalog() -> None:
    async def it_is_not_asked_again_until_its_ttl_passes(tmp_path: Path, catalog: Catalog) -> None:
        _, cache, provider = await registered(tmp_path / "skyward.sqlite")
        catalog.offers = ()

        await read(cache)
        await read(cache)
        await attempted_ago(provider, catalog.ttl - timedelta(seconds=5))
        await read(cache)
        assert catalog.asked == 1, "nothing to offer is an answer, and it holds for the ttl"

        await attempted_ago(provider, catalog.ttl + timedelta(seconds=1))
        await read(cache)
        assert catalog.asked == 2


def describe_a_provider_updated() -> None:
    async def it_has_its_catalog_fetched_on_the_next_read(tmp_path: Path, catalog: Catalog) -> None:
        providers, cache, _ = await registered(tmp_path / "skyward.sqlite")
        await read(cache)
        assert catalog.asked == 1

        catalog.offers = ("stub-b",)
        await providers.update("stubby", ProviderCreate(name="stubby", kind="stub", credentials={}, config={"region": "elsewhere"}))
        offers = await read(cache)

        assert catalog.asked == 2, "the catalog cached was the old account's"
        assert [offer.id for offer in offers.items] == ["stub-b"]


def describe_reading_the_catalog() -> None:
    async def it_orders_by_what_one_accelerator_costs(tmp_path: Path, catalog: Catalog) -> None:
        catalog.offers = ("one", "two", "three")
        _, cache, _ = await registered(tmp_path / "skyward.sqlite")

        page = await read(cache)

        assert [offer.id for offer in page.items] == ["three", "two", "one"], "one price over three accelerators is a third of it each"

    async def it_orders_by_what_is_available_when_asked_to(tmp_path: Path, catalog: Catalog) -> None:
        catalog.offers = ("one", "two", "three")
        _, cache, _ = await registered(tmp_path / "skyward.sqlite")

        page = await read(cache, sort="available")

        assert [offer.id for offer in page.items] == ["one", "two", "three"]

    async def it_lists_only_what_can_be_had_at_a_spot_price_when_asked_to(tmp_path: Path, catalog: Catalog) -> None:
        catalog.offers = ("fixed", "bidden")
        catalog.spot = frozenset({"bidden"})
        _, cache, _ = await registered(tmp_path / "skyward.sqlite")

        page = await read(cache, spot=True)

        assert [offer.id for offer in page.items] == ["bidden"]

    async def a_page_says_how_many_matched_rather_than_how_many_it_carries(tmp_path: Path, catalog: Catalog) -> None:
        catalog.offers = ("one", "two", "three")
        _, cache, _ = await registered(tmp_path / "skyward.sqlite")

        page = await read(cache, limit=2)

        assert [offer.id for offer in page.items] == ["three", "two"] and page.total == 3
