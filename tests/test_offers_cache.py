import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from litestar.testing import AsyncTestClient

from skyward.shared.errors import CapabilityMismatchError, NotFoundError
from skyward.server.persistence.db import connect
from skyward.server.persistence.offers import OfferCache
from skyward.server.persistence.providers import ProviderStore
from skyward.server.persistence.tables import OfferRow, ProviderRow
from skyward.shared.schemas import ProviderCreate
from skyward.providers.registry import REGISTRY
from skyward.providers.fake import CATALOG, FakeProvider
from skyward.server.http.app import create_app, with_real


@pytest.fixture
async def store(tmp_path):
    await connect(tmp_path / "test.sqlite")
    yield ProviderStore()


@pytest.fixture
async def cache(store):
    return OfferCache(store)


async def test_registers_a_provider_and_lists_it(store):
    provider = await store.create(ProviderCreate(name="local", kind="fake", config={"region": "lab-1"}))

    assert provider.id.startswith("prv_")
    assert provider.kind == "fake"
    assert provider.offers_ttl_seconds == int(FakeProvider.offers_ttl.total_seconds())
    assert (await store.get("local")).id == provider.id
    assert (await store.list()).items[0].id == provider.id


async def test_credentials_are_stored_but_never_returned(store):
    await store.create(ProviderCreate(name="vast", kind="vastai", credentials={"api_key": "sk-secret"}))

    provider = await store.get("vast")
    assert "sk-secret" not in str(provider)

    row = await ProviderRow.objects().output(load_json=True).where(ProviderRow.name == "vast").first()
    assert row.credentials == {"api_key": "sk-secret"}


async def test_missing_credentials_are_refused_before_the_row_is_written(store):
    with pytest.raises(CapabilityMismatchError):
        await store.create(ProviderCreate(name="vast", kind="vastai"))

    assert await ProviderRow.count().where(ProviderRow.name == "vast") == 0


async def test_offers_are_fetched_on_miss_and_served_from_cache_after(store, cache, monkeypatch):
    await store.create(ProviderCreate(name="local", kind="fake"))

    calls = 0
    original = FakeProvider.offers

    def counting(self):
        nonlocal calls
        calls += 1
        return original(self)

    monkeypatch.setattr(FakeProvider, "offers", counting)

    first = await cache.list(None, None, None, None, None, None, refresh=False)
    second = await cache.list(None, None, None, None, None, None, refresh=False)

    assert calls == 1, "second call must be served from cache"
    assert len(first.items) == len(second.items) == len(CATALOG)


async def test_expired_offers_are_refetched(store, cache):
    provider = await store.create(ProviderCreate(name="local", kind="fake"))
    await cache.list(None, None, None, None, None, None, refresh=False)

    await OfferRow.update({OfferRow.expires_at: datetime.now(UTC) - timedelta(seconds=1)}).where(
        OfferRow.provider_id == provider.id,
    ).run()

    offers = await cache.list(None, None, None, None, None, None, refresh=False)
    assert all(offer.expires_at > datetime.now(UTC) for offer in offers.items)


async def test_a_refresh_that_fails_serves_stale_offers_and_records_the_error(store, cache, monkeypatch):
    provider = await store.create(ProviderCreate(name="local", kind="fake"))
    await cache.list(None, None, None, None, None, None, refresh=False)

    def exploding(self):
        raise RuntimeError("provider is down")

    monkeypatch.setattr(FakeProvider, "offers", exploding)

    offers = await cache.list(None, None, None, None, None, None, refresh=True)

    assert len(offers.items) == len(CATALOG), "a dead provider must degrade the answer, not erase the catalog"
    assert (await store.get(provider.id)).last_error is not None


async def test_a_refresh_drops_offers_that_vanished_from_the_catalog(store, cache, monkeypatch):
    await store.create(ProviderCreate(name="local", kind="fake"))
    await cache.list(None, None, None, None, None, None, refresh=False)

    monkeypatch.setattr("skyward.providers.fake.CATALOG", (("a100", 1, 12, 85.0, 1.10, 2.20),))

    offers = await cache.list(None, None, None, None, None, None, refresh=True)
    assert len(offers.items) == 1, "an offer that left the provider's catalog must leave the cache"


async def test_concurrent_reads_refresh_once(store, cache, monkeypatch):
    await store.create(ProviderCreate(name="local", kind="fake"))

    calls = 0
    original = FakeProvider.offers

    def counting(self):
        nonlocal calls
        calls += 1
        return original(self)

    monkeypatch.setattr(FakeProvider, "offers", counting)

    await asyncio.gather(*(cache.list(None, None, None, None, None, None, refresh=False) for _ in range(8)))
    assert calls == 1, "eight concurrent readers must not hammer the provider eight times"


async def test_filters_run_against_the_cache(store, cache):
    await store.create(ProviderCreate(name="local", kind="fake"))

    a100s = await cache.list(None, None, "a100", None, None, None, refresh=False)
    assert {offer.accelerator for offer in a100s.items} == {"a100"}

    big = await cache.list(None, None, None, 8, None, None, refresh=False)
    assert all(offer.accelerator_count >= 8 for offer in big.items)

    cheap = await cache.list(None, None, None, None, None, 1.0, refresh=False)
    assert all(offer.price <= 1.0 for offer in cheap.items)

    ordered = await cache.list(None, None, None, None, None, None, refresh=False)
    prices = [offer.price for offer in ordered.items]
    assert prices == sorted(prices)


async def test_a_spot_only_offer_is_still_comparable_on_price(store, cache):
    await store.create(ProviderCreate(name="local", kind="fake"))

    offers = await cache.list(None, None, "h100", None, None, 7.0, refresh=False)
    spot_only = [offer for offer in offers.items if offer.on_demand_price is None]

    assert spot_only, "an offer with no on-demand price must survive a budget filter on its spot price"
    assert all(offer.price == offer.spot_price for offer in spot_only)
    assert offers.items[0].price == min(offer.price for offer in offers.items)


async def test_a_catalog_too_big_for_one_insert_still_lands(store, cache, monkeypatch):
    """Vultr publishes ~3000 offers. SQLite caps a statement at 32766 bound variables."""
    big = tuple(("h100", count, 8, 64.0, 1.0, 2.0) for count in range(1, 2001))
    monkeypatch.setattr("skyward.providers.fake.CATALOG", big)
    await store.create(ProviderCreate(name="local", kind="fake"))

    offers = await cache.list(None, None, None, None, None, None, refresh=False)
    assert len(offers.items) == 2000


async def test_deleting_a_provider_drops_its_offers(store, cache):
    provider = await store.create(ProviderCreate(name="local", kind="fake"))
    await cache.list(None, None, None, None, None, None, refresh=False)

    await store.delete("local")

    assert await OfferRow.count().where(OfferRow.provider_id == provider.id) == 0
    with pytest.raises(NotFoundError):
        await store.get("local")


async def test_the_whole_path_over_http(tmp_path):
    await connect(tmp_path / "http.sqlite")
    store = ProviderStore()
    services = with_real(providers=store, offers=OfferCache(store))

    async with AsyncTestClient(app=create_app(services)) as client:
        kinds = await client.get("/v1/provider-kinds")
        assert "fake" in {kind["kind"] for kind in kinds.json()}

        created = await client.post("/v1/providers", json={"name": "local", "kind": "fake", "config": {"region": "lab-1"}})
        assert created.status_code == 201
        assert "credentials" not in created.json()

        offers = await client.get("/v1/offers", params={"accelerator": "h100", "min_count": 8})
        assert offers.status_code == 200
        items = offers.json()["items"]
        assert len(items) == 1
        assert items[0]["accelerator_count"] == 8
        assert items[0]["region"] == "lab-1"

        provider = await client.get("/v1/providers/local")
        assert provider.json()["offers_count"] == len(CATALOG)
        assert provider.json()["offers_fetched_at"] is not None

        assert (await client.delete("/v1/providers/local")).status_code == 204
        assert (await client.get("/v1/providers/local")).status_code == 404


def test_every_registered_kind_satisfies_the_contract():
    for kind, adapter in REGISTRY.items():
        assert adapter.kind == kind
        assert adapter.offers_ttl.total_seconds() > 0
        assert callable(adapter.create)
