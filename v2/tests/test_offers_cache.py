import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from litestar.testing import AsyncTestClient

from skyward2.application.errors import CapabilityMismatchError, NotFoundError
from skyward2.persistence.db import connect
from skyward2.persistence.offers import OfferCache
from skyward2.persistence.providers import ProviderStore
from skyward2.persistence.tables import OfferRow, ProviderRow
from skyward2.protocol.schemas import ProviderCreate
from skyward2.providers import REGISTRY
from skyward2.providers.fake import FakeProvider
from skyward2.server.app import create_app, with_real


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

    first = await cache.list(None, None, None, None, None, refresh=False)
    second = await cache.list(None, None, None, None, None, refresh=False)

    assert calls == 1, "second call must be served from cache"
    assert len(first.items) == len(second.items) == 5


async def test_expired_offers_are_refetched(store, cache):
    provider = await store.create(ProviderCreate(name="local", kind="fake"))
    await cache.list(None, None, None, None, None, refresh=False)

    await OfferRow.update({OfferRow.expires_at: datetime.now(UTC) - timedelta(seconds=1)}).where(
        OfferRow.provider_id == provider.id,
    ).run()

    offers = await cache.list(None, None, None, None, None, refresh=False)
    assert all(offer.expires_at > datetime.now(UTC) for offer in offers.items)


async def test_a_refresh_that_fails_serves_stale_offers_and_records_the_error(store, cache, monkeypatch):
    provider = await store.create(ProviderCreate(name="local", kind="fake"))
    await cache.list(None, None, None, None, None, refresh=False)

    def exploding(self):
        raise RuntimeError("provider is down")

    monkeypatch.setattr(FakeProvider, "offers", exploding)

    offers = await cache.list(None, None, None, None, None, refresh=True)

    assert len(offers.items) == 5, "a dead provider must degrade the answer, not erase the catalog"
    assert (await store.get(provider.id)).last_error is not None


async def test_a_refresh_drops_offers_that_vanished_from_the_catalog(store, cache, monkeypatch):
    await store.create(ProviderCreate(name="local", kind="fake"))
    await cache.list(None, None, None, None, None, refresh=False)

    monkeypatch.setattr("skyward2.providers.fake.CATALOG", (("a100", 1, 12, 85.0, 1.10, 2.20),))

    offers = await cache.list(None, None, None, None, None, refresh=True)
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

    await asyncio.gather(*(cache.list(None, None, None, None, None, refresh=False) for _ in range(8)))
    assert calls == 1, "eight concurrent readers must not hammer the provider eight times"


async def test_filters_run_against_the_cache(store, cache):
    await store.create(ProviderCreate(name="local", kind="fake"))

    a100s = await cache.list(None, None, "a100", None, None, refresh=False)
    assert {offer.accelerator for offer in a100s.items} == {"a100"}

    big = await cache.list(None, None, None, 8, None, refresh=False)
    assert all(offer.accelerator_count >= 8 for offer in big.items)

    cheap = await cache.list(None, None, None, None, 1.0, refresh=False)
    assert all(offer.on_demand_price <= 1.0 for offer in cheap.items)

    ordered = await cache.list(None, None, None, None, None, refresh=False)
    prices = [offer.on_demand_price for offer in ordered.items]
    assert prices == sorted(prices)


async def test_deleting_a_provider_drops_its_offers(store, cache):
    provider = await store.create(ProviderCreate(name="local", kind="fake"))
    await cache.list(None, None, None, None, None, refresh=False)

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
        assert provider.json()["offers_count"] == 5
        assert provider.json()["offers_fetched_at"] is not None

        assert (await client.delete("/v1/providers/local")).status_code == 204
        assert (await client.get("/v1/providers/local")).status_code == 404


def test_every_registered_kind_satisfies_the_contract():
    for kind, adapter in REGISTRY.items():
        assert adapter.kind == kind
        assert adapter.offers_ttl.total_seconds() > 0
        assert callable(adapter.create)
