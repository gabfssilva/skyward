from __future__ import annotations

import sqlite3

import pytest

from skyward.accelerators.spec import Accelerator
from skyward.core.model import InstanceType, Offer
from skyward.infra.threaded import ThreadPoolRunner
from skyward.offers.conversion import _offer_from_runtime
from skyward.offers.repository import OfferRepository, _SCHEMA


def _make_offer(
    provider: str = "aws",
    itype: str = "p4d.24xlarge",
    gpu_name: str = "A100",
    gpu_count: int = 8,
    vcpus: float = 96,
    memory_gb: float = 1024,
    spot: float | None = 11.91,
    on_demand: float | None = 21.96,
) -> Offer:
    return Offer(
        id=f"{provider}-us-east-1-{itype}",
        instance_type=InstanceType(
            name=itype,
            accelerator=Accelerator(name=gpu_name, count=gpu_count),
            vcpus=vcpus,
            memory_gb=memory_gb,
            architecture="x86_64",
            specific=None,
        ),
        spot_price=spot,
        on_demand_price=on_demand,
        billing_unit="second",
        specific=None,
    )


class TestOfferFromRuntime:
    def test_roundtrip_preserves_gpu(self) -> None:
        offer = _make_offer()
        feed_offer = _offer_from_runtime(offer, "aws")
        assert feed_offer.spec.accelerator.name == "A100"
        assert feed_offer.accelerator_count == 8

    def test_roundtrip_preserves_pricing(self) -> None:
        offer = _make_offer(spot=5.0, on_demand=10.0)
        feed_offer = _offer_from_runtime(offer, "aws")
        assert feed_offer.spot_price == 5.0
        assert feed_offer.on_demand_price == 10.0

    def test_roundtrip_preserves_instance_type(self) -> None:
        offer = _make_offer(itype="g4dn.xlarge")
        feed_offer = _offer_from_runtime(offer, "aws")
        assert feed_offer.instance_type == "g4dn.xlarge"

    def test_none_pricing(self) -> None:
        offer = _make_offer(spot=None, on_demand=None)
        feed_offer = _offer_from_runtime(offer, "aws")
        assert feed_offer.spot_price is None
        assert feed_offer.on_demand_price is None

    def test_no_accelerator(self) -> None:
        offer = Offer(
            id="aws-us-east-1-c5.xlarge",
            instance_type=InstanceType(
                name="c5.xlarge",
                accelerator=None,
                vcpus=4,
                memory_gb=8,
                architecture="x86_64",
                specific=None,
            ),
            spot_price=0.07,
            on_demand_price=0.17,
            billing_unit="second",
            specific=None,
        )
        feed_offer = _offer_from_runtime(offer, "aws")
        assert feed_offer.spec.accelerator.name == ""
        assert feed_offer.accelerator_count == 0


def _empty_repo(*providers: FakeProvider) -> OfferRepository:
    db = sqlite3.connect(":memory:", check_same_thread=False)
    db.executescript(_SCHEMA)
    db.row_factory = sqlite3.Row
    runner = ThreadPoolRunner(workers=1)
    provider_map = {p.name: p for p in providers}
    return OfferRepository(db, providers=provider_map, runner=runner)


class FakeProvider:
    name = "fake"

    def __init__(self, offers: list[Offer] | None = None) -> None:
        self._offers = offers or []
        self.call_count = 0

    async def offers(self) -> object:
        self.call_count += 1
        for o in self._offers:
            yield o


def _fake_offer(
    itype: str = "gpu.large",
    gpu: str = "A100",
    spot: float = 10.0,
) -> Offer:
    return Offer(
        id=f"fake-us-east-1-{itype}",
        instance_type=InstanceType(
            name=itype,
            accelerator=Accelerator(name=gpu, count=1),
            vcpus=8,
            memory_gb=64,
            architecture="x86_64",
            specific=None,
        ),
        spot_price=spot,
        on_demand_price=spot * 2,
        billing_unit="hour",
        specific=None,
    )


class TestOnDemandFetch:
    @pytest.mark.asyncio
    async def test_empty_catalog_triggers_fetch(self) -> None:
        provider = FakeProvider([_fake_offer()])
        repo = _empty_repo(provider)
        results = await repo.accelerator("A100").provider("fake").all()
        assert len(results) >= 1
        assert results[0].accelerator_name == "A100"
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_cached_results_skip_fetch(self) -> None:
        provider = FakeProvider([_fake_offer()])
        repo = _empty_repo(provider)
        await repo.accelerator("A100").provider("fake").all()
        assert provider.call_count == 1
        results = await repo.accelerator("A100").provider("fake").all()
        assert len(results) >= 1
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_provider_filter_scopes_fetch(self) -> None:
        fake1 = FakeProvider([_fake_offer(gpu="A100")])
        fake1.name = "provider_a"
        fake2 = FakeProvider([_fake_offer(gpu="H100")])
        fake2.name = "provider_b"
        repo = _empty_repo(fake1, fake2)
        await repo.accelerator("A100").provider("provider_a").all()
        assert fake1.call_count == 1
        assert fake2.call_count == 0

    @pytest.mark.asyncio
    async def test_no_provider_returns_empty(self) -> None:
        repo = _empty_repo()
        results = await repo.accelerator("A100").all()
        assert results == []

    @pytest.mark.asyncio
    async def test_cheapest_triggers_fetch(self) -> None:
        provider = FakeProvider([
            _fake_offer(itype="cheap", spot=1.0),
            _fake_offer(itype="expensive", spot=99.0),
        ])
        repo = _empty_repo(provider)
        result = await repo.provider("fake").cheapest()
        assert result is not None
        assert result.spot_price == 1.0
