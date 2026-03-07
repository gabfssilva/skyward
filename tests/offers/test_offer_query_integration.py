"""Integration tests for OfferRepository DSL — real API calls via on-demand fetch.

Run all:    uv run pytest tests/offers/test_offer_query_integration.py -v -m integration
Run one:    uv run pytest tests/offers/test_offer_query_integration.py -k aws -v -m integration
"""

from __future__ import annotations

import pytest

from skyward.offers import OfferRepository


# ── AWS ──────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.aws
@pytest.mark.asyncio
async def test_aws_a100_cheapest() -> None:
    from skyward.providers.aws.config import AWS

    provider = await AWS().create_provider()
    repo = await OfferRepository.create(providers=[provider])

    offer = await repo.provider("aws").accelerator("A100").cheapest()
    assert offer is not None
    assert offer.accelerator_name == "A100"
    assert offer.provider == "aws"
    assert offer.on_demand_price is not None or offer.spot_price is not None


@pytest.mark.integration
@pytest.mark.aws
@pytest.mark.asyncio
async def test_aws_t4_spot() -> None:
    from skyward.providers.aws.config import AWS

    provider = await AWS().create_provider()
    repo = await OfferRepository.create(providers=[provider])

    offers = await repo.provider("aws").accelerator("T4").spot().cheapest(5)
    assert len(offers) >= 1
    for o in offers:
        assert o.accelerator_name == "T4"
        assert o.spot_price is not None


@pytest.mark.integration
@pytest.mark.aws
@pytest.mark.asyncio
async def test_aws_h100_8gpu() -> None:
    from skyward.providers.aws.config import AWS

    provider = await AWS().create_provider()
    repo = await OfferRepository.create(providers=[provider])

    offers = await repo.provider("aws").accelerator("H100").accelerator_count(8).all()
    assert len(offers) >= 1
    for o in offers:
        assert o.accelerator_name == "H100"
        assert o.accelerator_count >= 8


@pytest.mark.integration
@pytest.mark.aws
@pytest.mark.asyncio
async def test_aws_a10g_on_demand() -> None:
    from skyward.providers.aws.config import AWS

    provider = await AWS().create_provider()
    repo = await OfferRepository.create(providers=[provider])

    offer = await repo.provider("aws").accelerator("A10G").on_demand().cheapest()
    assert offer is not None
    assert offer.accelerator_name == "A10G"
    assert offer.on_demand_price is not None


# ── GCP ──────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.gcp
@pytest.mark.asyncio
async def test_gcp_l4_cheapest() -> None:
    from skyward.providers.gcp.config import GCP

    provider = await GCP().create_provider()
    repo = await OfferRepository.create(providers=[provider])

    offers = await repo.provider("gcp").accelerator("L4").cheapest(3)
    assert len(offers) >= 1
    for o in offers:
        assert o.accelerator_name == "L4"
        assert o.provider == "gcp"


@pytest.mark.integration
@pytest.mark.gcp
@pytest.mark.asyncio
async def test_gcp_t4_all() -> None:
    from skyward.providers.gcp.config import GCP

    provider = await GCP().create_provider()
    repo = await OfferRepository.create(providers=[provider])

    offers = await repo.provider("gcp").accelerator("T4").all()
    assert len(offers) >= 1
    for o in offers:
        assert o.accelerator_name == "T4"


# ── VastAI ───────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.vastai
@pytest.mark.asyncio
async def test_vastai_a100_vram80() -> None:
    from skyward.providers.vastai.config import VastAI

    provider = await VastAI().create_provider()
    repo = await OfferRepository.create(providers=[provider])

    offer = await repo.provider("vastai").accelerator("A100").accelerator_memory(80).cheapest()
    assert offer is not None
    assert offer.accelerator_name == "A100"
    assert offer.accelerator_memory_gb >= 80


@pytest.mark.integration
@pytest.mark.vastai
@pytest.mark.asyncio
async def test_vastai_h100_cheapest() -> None:
    from skyward.providers.vastai.config import VastAI

    provider = await VastAI().create_provider()
    repo = await OfferRepository.create(providers=[provider])

    offer = await repo.provider("vastai").accelerator("H100").cheapest()
    assert offer is not None
    assert offer.accelerator_name == "H100"


# ── RunPod ───────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.runpod
@pytest.mark.asyncio
async def test_runpod_a100_cheapest() -> None:
    from skyward.providers.runpod.config import RunPod

    provider = await RunPod().create_provider()
    repo = await OfferRepository.create(providers=[provider])

    offer = await repo.provider("runpod").accelerator("A100").cheapest()
    assert offer is not None
    assert offer.accelerator_name == "A100"


@pytest.mark.integration
@pytest.mark.runpod
@pytest.mark.asyncio
async def test_runpod_h100_all() -> None:
    from skyward.providers.runpod.config import RunPod

    provider = await RunPod().create_provider()
    repo = await OfferRepository.create(providers=[provider])

    offers = await repo.provider("runpod").accelerator("H100").all()
    assert len(offers) >= 1
    for o in offers:
        assert o.accelerator_name == "H100"


# ── Hyperstack ───────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.hyperstack
@pytest.mark.asyncio
async def test_hyperstack_a100_cheapest() -> None:
    from skyward.providers.hyperstack.config import Hyperstack

    provider = await Hyperstack().create_provider()
    repo = await OfferRepository.create(providers=[provider])

    offer = await repo.provider("hyperstack").accelerator("A100").cheapest()
    assert offer is not None
    assert offer.accelerator_name == "A100"


# ── TensorDock ───────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.tensordock
@pytest.mark.asyncio
async def test_tensordock_a100_cheapest() -> None:
    from skyward.providers.tensordock.config import TensorDock

    provider = await TensorDock().create_provider()
    repo = await OfferRepository.create(providers=[provider])

    offer = await repo.provider("tensordock").accelerator("A100").cheapest()
    assert offer is not None
    assert offer.accelerator_name == "A100"


@pytest.mark.integration
@pytest.mark.tensordock
@pytest.mark.asyncio
async def test_tensordock_h100_cheapest() -> None:
    from skyward.providers.tensordock.config import TensorDock

    provider = await TensorDock().create_provider()
    repo = await OfferRepository.create(providers=[provider])

    offer = await repo.provider("tensordock").accelerator("H100").cheapest()
    assert offer is not None
    assert offer.accelerator_name == "H100"


# ── Verda ────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.verda
@pytest.mark.asyncio
async def test_verda_a100_cheapest() -> None:
    from skyward.providers.verda.config import Verda

    provider = await Verda().create_provider()
    repo = await OfferRepository.create(providers=[provider])

    offer = await repo.provider("verda").accelerator("A100").cheapest()
    assert offer is not None
    assert offer.accelerator_name == "A100"


@pytest.mark.integration
@pytest.mark.verda
@pytest.mark.asyncio
async def test_verda_h100_cheapest() -> None:
    from skyward.providers.verda.config import Verda

    provider = await Verda().create_provider()
    repo = await OfferRepository.create(providers=[provider])

    offer = await repo.provider("verda").accelerator("H100").cheapest()
    assert offer is not None
    assert offer.accelerator_name == "H100"


# ── CPU-only ─────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.aws
@pytest.mark.asyncio
async def test_aws_cpu_only_cheapest() -> None:
    from skyward.providers.aws.config import AWS

    provider = await AWS().create_provider()
    repo = await OfferRepository.create(providers=[provider])

    offer = await repo.provider("aws").cpu_only().vcpus(4).memory(8).cheapest()
    assert offer is not None
    assert offer.accelerator_name == ""
    assert offer.accelerator_count == 0
    assert offer.vcpus == 4
    assert offer.memory_gb == 8
    assert offer.on_demand_price is not None or offer.spot_price is not None


@pytest.mark.integration
@pytest.mark.aws
@pytest.mark.asyncio
async def test_aws_cpu_only_range() -> None:
    from skyward.providers.aws.config import AWS

    provider = await AWS().create_provider()
    repo = await OfferRepository.create(providers=[provider])

    offers = await repo.provider("aws").cpu_only().vcpus(4, 8).memory(8, 32).all()
    assert len(offers) >= 1
    for o in offers:
        assert o.accelerator_name == ""
        assert o.accelerator_count == 0
        assert 4 <= o.vcpus <= 8
        assert 8 <= o.memory_gb <= 32


@pytest.mark.integration
@pytest.mark.gcp
@pytest.mark.asyncio
async def test_gcp_cpu_only_cheapest() -> None:
    from skyward.providers.gcp.config import GCP

    provider = await GCP().create_provider()
    repo = await OfferRepository.create(providers=[provider])

    offer = await repo.provider("gcp").cpu_only().vcpus(4).memory(16).cheapest()
    assert offer is not None
    assert offer.accelerator_name == ""
    assert offer.accelerator_count == 0
    assert offer.vcpus == 4


# ── Cross-provider ───────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.vastai
@pytest.mark.runpod
@pytest.mark.asyncio
async def test_multi_provider_a100_cheapest() -> None:
    from skyward.providers.runpod.config import RunPod
    from skyward.providers.vastai.config import VastAI

    vastai = await VastAI().create_provider()
    runpod = await RunPod().create_provider()
    repo = await OfferRepository.create(providers=[vastai, runpod])

    offers = await repo.accelerator("A100").cheapest(5)
    assert len(offers) >= 1
    providers = {o.provider for o in offers}
    assert providers <= {"vastai", "runpod"}
