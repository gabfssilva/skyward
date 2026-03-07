from __future__ import annotations

import json
import sqlite3

import pytest

from skyward.infra.threaded import ThreadPoolRunner
from skyward.offers.query import OfferQuery
from skyward.offers.repository import OfferRepository, _SCHEMA


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ACCELERATORS = [
    {"id": "acc-a100", "name": "A100", "vram": 80.0, "manufacturer": "NVIDIA", "architecture": "Ampere", "cuda_min": "11.0", "cuda_max": "12.6"},
    {"id": "acc-h100", "name": "H100", "vram": 80.0, "manufacturer": "NVIDIA", "architecture": "Hopper", "cuda_min": "11.8", "cuda_max": "12.6"},
    {"id": "acc-t4", "name": "T4", "vram": 16.0, "manufacturer": "NVIDIA", "architecture": "Turing", "cuda_min": "10.0", "cuda_max": "12.6"},
    {"id": "acc-none", "name": "", "vram": 0.0, "manufacturer": "", "architecture": "", "cuda_min": "", "cuda_max": ""},
]

_SPECS = [
    {"id": "spec-a100-96-1024", "accelerator_id": "acc-a100", "vcpus": 96, "memory_gb": 1024},
    {"id": "spec-h100-192-2048", "accelerator_id": "acc-h100", "vcpus": 192, "memory_gb": 2048},
    {"id": "spec-t4-8-32", "accelerator_id": "acc-t4", "vcpus": 8, "memory_gb": 32},
    {"id": "spec-cpu-4-16", "accelerator_id": "acc-none", "vcpus": 4, "memory_gb": 16},
    {"id": "spec-cpu-8-32", "accelerator_id": "acc-none", "vcpus": 8, "memory_gb": 32},
    {"id": "spec-cpu-96-192", "accelerator_id": "acc-none", "vcpus": 96, "memory_gb": 192},
]

_OFFERS = [
    # GPU offers
    {"provider": "aws", "spec_id": "spec-a100-96-1024", "accelerator_count": 8, "instance_type": "p4d.24xlarge", "region": "us-east-1", "spot_price": 11.91, "on_demand_price": 21.96, "billing_unit": "second", "specific": json.dumps({"architecture": "x86_64"})},
    {"provider": "aws", "spec_id": "spec-h100-192-2048", "accelerator_count": 8, "instance_type": "p5.48xlarge", "region": "us-east-1", "spot_price": 19.48, "on_demand_price": 55.04, "billing_unit": "second", "specific": json.dumps({"architecture": "x86_64"})},
    {"provider": "aws", "spec_id": "spec-t4-8-32", "accelerator_count": 1, "instance_type": "g4dn.2xlarge", "region": "us-east-1", "spot_price": 0.31, "on_demand_price": 0.75, "billing_unit": "second", "specific": json.dumps({"architecture": "x86_64"})},
    {"provider": "aws", "spec_id": "spec-h100-192-2048", "accelerator_count": 8, "instance_type": "p5.48xlarge", "region": "eu-west-1", "spot_price": 22.10, "on_demand_price": 60.50, "billing_unit": "second", "specific": json.dumps({"architecture": "x86_64"})},
    # CPU-only offers
    {"provider": "aws", "spec_id": "spec-cpu-4-16", "accelerator_count": 0, "instance_type": "c5.xlarge", "region": "us-east-1", "spot_price": 0.07, "on_demand_price": 0.17, "billing_unit": "second", "specific": json.dumps({"architecture": "x86_64"})},
    {"provider": "aws", "spec_id": "spec-cpu-8-32", "accelerator_count": 0, "instance_type": "m5.2xlarge", "region": "us-east-1", "spot_price": 0.15, "on_demand_price": 0.38, "billing_unit": "second", "specific": json.dumps({"architecture": "x86_64"})},
    {"provider": "aws", "spec_id": "spec-cpu-96-192", "accelerator_count": 0, "instance_type": "c5.24xlarge", "region": "us-east-1", "spot_price": 1.62, "on_demand_price": 4.08, "billing_unit": "second", "specific": json.dumps({"architecture": "x86_64"})},
    {"provider": "aws", "spec_id": "spec-cpu-8-32", "accelerator_count": 0, "instance_type": "m5.2xlarge", "region": "eu-west-1", "spot_price": 0.18, "on_demand_price": 0.42, "billing_unit": "second", "specific": json.dumps({"architecture": "x86_64"})},
    # CPU-only, spot only (no on-demand)
    {"provider": "aws", "spec_id": "spec-cpu-4-16", "accelerator_count": 0, "instance_type": "c5.xlarge-spot-only", "region": "us-east-1", "spot_price": 0.05, "on_demand_price": None, "billing_unit": "second", "specific": None},
]


def _build_repo() -> OfferRepository:
    db = sqlite3.connect(":memory:", check_same_thread=False)
    db.executescript(_SCHEMA)
    db.row_factory = sqlite3.Row
    for a in _ACCELERATORS:
        db.execute(
            "INSERT OR IGNORE INTO accelerators VALUES (?, ?, ?, ?, ?, ?, ?)",
            (a["id"], a["name"], a["vram"], a["manufacturer"], a["architecture"], a["cuda_min"], a["cuda_max"]),
        )
    for s in _SPECS:
        db.execute(
            "INSERT OR IGNORE INTO specs VALUES (?, ?, ?, ?, ?)",
            (s["id"], s["accelerator_id"], s["vcpus"], s["memory_gb"], s.get("cpu_architecture", "x86_64")),
        )
    for o in _OFFERS:
        db.execute(
            "INSERT INTO offers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (o["provider"], o["spec_id"], o["accelerator_count"], o["instance_type"],
             o["region"], o["spot_price"], o["on_demand_price"], o["billing_unit"], o.get("specific")),
        )
    db.commit()
    runner = ThreadPoolRunner(workers=1)
    return OfferRepository(db, providers={}, runner=runner)


@pytest.fixture
def repo() -> OfferRepository:
    return _build_repo()


# ---------------------------------------------------------------------------
# CPU-only queries
# ---------------------------------------------------------------------------


class TestCPUOffers:
    @pytest.mark.asyncio
    async def test_cpu_offers_have_zero_accelerator_count(self, repo: OfferRepository) -> None:
        offers = await OfferQuery(repo._db, repo).where("accelerator_count = 0").all()
        assert len(offers) == 5
        for o in offers:
            assert o.accelerator_count == 0
            assert o.accelerator_name == ""
            assert o.accelerator_memory_gb == 0.0

    @pytest.mark.asyncio
    async def test_cpu_offers_filtered_by_vcpus_exact(self, repo: OfferRepository) -> None:
        offers = await OfferQuery(repo._db, repo).where("accelerator_count = 0").vcpus(8).all()
        assert all(o.vcpus == 8 for o in offers)
        instance_types = {o.instance_type for o in offers}
        assert "c5.xlarge" not in instance_types
        assert "m5.2xlarge" in instance_types

    @pytest.mark.asyncio
    async def test_cpu_offers_filtered_by_vcpus_range(self, repo: OfferRepository) -> None:
        offers = await OfferQuery(repo._db, repo).where("accelerator_count = 0").vcpus(8, 96).all()
        assert all(8 <= o.vcpus <= 96 for o in offers)
        instance_types = {o.instance_type for o in offers}
        assert "c5.xlarge" not in instance_types
        assert "m5.2xlarge" in instance_types
        assert "c5.24xlarge" in instance_types

    @pytest.mark.asyncio
    async def test_cpu_offers_filtered_by_memory_exact(self, repo: OfferRepository) -> None:
        offers = await OfferQuery(repo._db, repo).where("accelerator_count = 0").memory(192).all()
        assert len(offers) == 1
        assert offers[0].instance_type == "c5.24xlarge"

    @pytest.mark.asyncio
    async def test_cpu_offers_filtered_by_memory_range(self, repo: OfferRepository) -> None:
        offers = await OfferQuery(repo._db, repo).where("accelerator_count = 0").memory(64, 192).all()
        assert len(offers) == 1
        assert offers[0].instance_type == "c5.24xlarge"

    @pytest.mark.asyncio
    async def test_cpu_cheapest(self, repo: OfferRepository) -> None:
        cheapest = await OfferQuery(repo._db, repo).where("accelerator_count = 0").cheapest()
        assert cheapest is not None
        assert cheapest.instance_type == "c5.xlarge-spot-only"
        assert cheapest.spot_price == 0.05

    @pytest.mark.asyncio
    async def test_cpu_cheapest_n(self, repo: OfferRepository) -> None:
        top3 = await OfferQuery(repo._db, repo).where("accelerator_count = 0").cheapest(3)
        assert len(top3) == 3
        prices: list[float] = [
            o.spot_price if o.spot_price is not None else (o.on_demand_price or 0.0)
            for o in top3
        ]
        assert prices == sorted(prices)

    @pytest.mark.asyncio
    async def test_cpu_spot_only(self, repo: OfferRepository) -> None:
        offers = await OfferQuery(repo._db, repo).where("accelerator_count = 0").spot().all()
        assert all(o.spot_price is not None for o in offers)
        assert len(offers) == 5

    @pytest.mark.asyncio
    async def test_cpu_on_demand_only(self, repo: OfferRepository) -> None:
        offers = await OfferQuery(repo._db, repo).where("accelerator_count = 0").on_demand().all()
        assert all(o.on_demand_price is not None for o in offers)
        assert "c5.xlarge-spot-only" not in {o.instance_type for o in offers}

    @pytest.mark.asyncio
    async def test_cpu_max_price(self, repo: OfferRepository) -> None:
        offers = await OfferQuery(repo._db, repo).where("accelerator_count = 0").max_price(0.10).all()
        for o in offers:
            effective = o.spot_price if o.spot_price is not None else o.on_demand_price
            assert effective is not None
            assert effective <= 0.10

    @pytest.mark.asyncio
    async def test_cpu_region_filter(self, repo: OfferRepository) -> None:
        offers = await OfferQuery(repo._db, repo).where("accelerator_count = 0").all()
        eu_offers = [o for o in offers if o.region == "eu-west-1"]
        assert len(eu_offers) == 1
        assert eu_offers[0].instance_type == "m5.2xlarge"

    @pytest.mark.asyncio
    async def test_cpu_region_via_query_builder(self, repo: OfferRepository) -> None:
        offers = await repo.region("eu-west-1").where("accelerator_count = 0").all()
        assert len(offers) == 1
        assert offers[0].instance_type == "m5.2xlarge"


# ---------------------------------------------------------------------------
# GPU queries (sanity check alongside CPU)
# ---------------------------------------------------------------------------


class TestGPUOffers:
    @pytest.mark.asyncio
    async def test_gpu_offers_have_positive_accelerator_count(self, repo: OfferRepository) -> None:
        offers = await OfferQuery(repo._db, repo).where("accelerator_count > 0").all()
        assert len(offers) == 4
        for o in offers:
            assert o.accelerator_count > 0
            assert o.accelerator_memory_gb > 0

    @pytest.mark.asyncio
    async def test_accelerator_filter(self, repo: OfferRepository) -> None:
        offers = await repo.accelerator("A100").all()
        assert len(offers) == 1
        assert offers[0].instance_type == "p4d.24xlarge"

    @pytest.mark.asyncio
    async def test_architecture_filter(self, repo: OfferRepository) -> None:
        offers = await repo.architecture("Hopper").all()
        assert len(offers) == 2
        assert all(o.accelerator_name == "H100" for o in offers)

    @pytest.mark.asyncio
    async def test_vram_filter(self, repo: OfferRepository) -> None:
        offers = await OfferQuery(repo._db, repo).accelerator_memory(80).all()
        assert all(o.accelerator_memory_gb >= 80 for o in offers)
        assert "g4dn.2xlarge" not in {o.instance_type for o in offers}


# ---------------------------------------------------------------------------
# Mixed queries (GPU + CPU together)
# ---------------------------------------------------------------------------


class TestMixedQueries:
    @pytest.mark.asyncio
    async def test_all_offers_returned(self, repo: OfferRepository) -> None:
        offers = await OfferQuery(repo._db, repo).all()
        assert len(offers) == len(_OFFERS)

    @pytest.mark.asyncio
    async def test_provider_filter_returns_all(self, repo: OfferRepository) -> None:
        offers = await repo.provider("aws").all()
        assert len(offers) == len(_OFFERS)

    @pytest.mark.asyncio
    async def test_cheapest_across_gpu_and_cpu(self, repo: OfferRepository) -> None:
        cheapest = await OfferQuery(repo._db, repo).cheapest()
        assert cheapest is not None
        assert cheapest.accelerator_count == 0

    @pytest.mark.asyncio
    async def test_region_filter_mixed(self, repo: OfferRepository) -> None:
        offers = await repo.region("eu-west-1").all()
        assert len(offers) == 2
        types = {o.instance_type for o in offers}
        assert types == {"p5.48xlarge", "m5.2xlarge"}

    @pytest.mark.asyncio
    async def test_vcpus_filter_mixed(self, repo: OfferRepository) -> None:
        offers = await OfferQuery(repo._db, repo).vcpus(96).all()
        types = {o.instance_type for o in offers}
        assert "p4d.24xlarge" in types
        assert "c5.24xlarge" in types
        assert "c5.xlarge" not in types

    @pytest.mark.asyncio
    async def test_max_price_mixed(self, repo: OfferRepository) -> None:
        offers = await OfferQuery(repo._db, repo).max_price(0.50).all()
        gpu = [o for o in offers if o.accelerator_count > 0]
        cpu = [o for o in offers if o.accelerator_count == 0]
        assert len(gpu) == 1
        assert gpu[0].instance_type == "g4dn.2xlarge"
        assert len(cpu) >= 3


# ---------------------------------------------------------------------------
# Raw SQL
# ---------------------------------------------------------------------------


class TestRawSQL:
    def test_raw_query_cpu_only(self, repo: OfferRepository) -> None:
        offers = repo.query(
            "SELECT * FROM catalog WHERE accelerator_count = 0 ORDER BY vcpus",
        )
        assert len(offers) == 5
        vcpus = [o.vcpus for o in offers]
        assert vcpus == sorted(vcpus)

    def test_raw_query_with_params(self, repo: OfferRepository) -> None:
        offers = repo.query(
            "SELECT * FROM catalog WHERE memory_gb >= ? AND accelerator_count = 0",
            (100,),
        )
        assert len(offers) == 1
        assert offers[0].instance_type == "c5.24xlarge"


# ---------------------------------------------------------------------------
# Extract filters
# ---------------------------------------------------------------------------


class TestExtractFilters:
    def test_extract_accelerator_filter(self, repo: OfferRepository) -> None:
        q = OfferQuery(repo._db, repo).accelerator("A100")
        assert q._extract_filters() == {"accelerator": "A100"}

    def test_extract_provider_filter(self, repo: OfferRepository) -> None:
        q = OfferQuery(repo._db, repo).provider("aws")
        assert q._extract_filters() == {"provider": "aws"}

    def test_extract_multiple_filters(self, repo: OfferRepository) -> None:
        q = OfferQuery(repo._db, repo).accelerator("H100").region("us-east-1").vcpus(96)
        filters = q._extract_filters()
        assert filters == {"accelerator": "H100", "region": "us-east-1", "vcpus": 96}

    def test_extract_ignores_unknown_clauses(self, repo: OfferRepository) -> None:
        q = OfferQuery(repo._db, repo).where("accelerator_count > 0").accelerator("A100")
        filters = q._extract_filters()
        assert filters == {"accelerator": "A100"}
