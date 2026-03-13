"""Integration tests for provider.offers() — real API calls, no provisioning.

Run all:    uv run pytest tests/offers/test_provider_offers.py -v -m integration
Run one:    uv run pytest tests/offers/test_provider_offers.py -k aws -v -m integration
"""

from __future__ import annotations

import pytest

from skyward.accelerators import Accelerator
from skyward.core.spec import Nodes, PoolSpec


def _gpu_spec(
    accelerator: str = "A100",
    region: str = "us-east-1",
    count: int = 1,
) -> PoolSpec:
    return PoolSpec(
        nodes=Nodes(min=1),
        accelerator=Accelerator(name=accelerator, count=count),
        region=region,
    )


def _cpu_spec(region: str = "us-east-1", vcpus: int = 4, memory_gb: float = 8) -> PoolSpec:
    return PoolSpec(
        nodes=Nodes(min=1),
        accelerator=None,
        region=region,
        vcpus=vcpus,
        memory_gb=memory_gb,
    )


def _assert_offers(
    offers: list,
    *,
    min_count: int = 1,
    require_pricing: bool = True,
    require_accelerator: bool = True,
) -> None:
    assert len(offers) >= min_count, f"Expected >= {min_count} offers, got {len(offers)}"
    for o in offers:
        assert o.instance_type.name, "Offer must have an instance type name"
        if require_pricing:
            assert (
                o.spot_price is not None or o.on_demand_price is not None
            ), f"Offer {o.id} has no pricing"
        if require_accelerator:
            assert o.instance_type.accelerator, f"Offer {o.id} has no accelerator"


# ── AWS ──────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.aws
@pytest.mark.asyncio
async def test_aws_offers_cpu_only() -> None:
    from skyward.providers.aws.config import AWS

    provider = await AWS().create_provider()
    offers = [o async for o in provider.offers(_cpu_spec("us-east-1", vcpus=4, memory_gb=8))]
    _assert_offers(offers, require_accelerator=False)


@pytest.mark.integration
@pytest.mark.aws
@pytest.mark.asyncio
async def test_aws_offers_t4() -> None:
    from skyward.providers.aws.config import AWS

    provider = await AWS().create_provider()
    offers = [o async for o in provider.offers(_gpu_spec("T4", "us-east-1"))]
    _assert_offers(offers)


@pytest.mark.integration
@pytest.mark.aws
@pytest.mark.asyncio
async def test_aws_offers_a100_8x() -> None:
    from skyward.providers.aws.config import AWS

    provider = await AWS().create_provider()
    offers = [o async for o in provider.offers(_gpu_spec("A100", "us-east-1", count=8))]
    _assert_offers(offers)


@pytest.mark.integration
@pytest.mark.aws
@pytest.mark.asyncio
async def test_aws_offers_a10g() -> None:
    from skyward.providers.aws.config import AWS

    provider = await AWS().create_provider()
    offers = [o async for o in provider.offers(_gpu_spec("A10G", "us-east-1"))]
    _assert_offers(offers)


@pytest.mark.integration
@pytest.mark.aws
@pytest.mark.asyncio
async def test_aws_offers_h100() -> None:
    from skyward.providers.aws.config import AWS

    provider = await AWS().create_provider()
    offers = [o async for o in provider.offers(_gpu_spec("H100", "us-east-1", count=8))]
    _assert_offers(offers)


# ── GCP ──────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.gcp
@pytest.mark.asyncio
async def test_gcp_offers_cpu_only() -> None:
    from skyward.providers.gcp.config import GCP

    provider = await GCP().create_provider()
    offers = [o async for o in provider.offers(_cpu_spec("us-central1", vcpus=4, memory_gb=16))]
    _assert_offers(offers, require_pricing=False, require_accelerator=False)


@pytest.mark.integration
@pytest.mark.gcp
@pytest.mark.asyncio
async def test_gcp_offers_l4() -> None:
    from skyward.providers.gcp.config import GCP

    provider = await GCP().create_provider()
    offers = [o async for o in provider.offers(_gpu_spec("L4", "us-central1"))]
    _assert_offers(offers, require_pricing=False)


@pytest.mark.integration
@pytest.mark.gcp
@pytest.mark.asyncio
async def test_gcp_offers_t4() -> None:
    from skyward.providers.gcp.config import GCP

    provider = await GCP().create_provider()
    offers = [o async for o in provider.offers(_gpu_spec("T4", "us-central1"))]
    _assert_offers(offers, require_pricing=False)


# ── VastAI ───────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.vastai
@pytest.mark.asyncio
async def test_vastai_offers_a100() -> None:
    from skyward.providers.vastai.config import VastAI

    provider = await VastAI().create_provider()
    offers = [o async for o in provider.offers(_gpu_spec("A100"))]
    _assert_offers(offers)


@pytest.mark.integration
@pytest.mark.vastai
@pytest.mark.asyncio
async def test_vastai_offers_h100() -> None:
    from skyward.providers.vastai.config import VastAI

    provider = await VastAI().create_provider()
    offers = [o async for o in provider.offers(_gpu_spec("H100"))]
    _assert_offers(offers)


# ── RunPod ───────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.runpod
@pytest.mark.asyncio
async def test_runpod_offers_a100() -> None:
    from skyward.providers.runpod.config import RunPod

    provider = await RunPod().create_provider()
    offers = [o async for o in provider.offers(_gpu_spec("A100"))]
    _assert_offers(offers)


@pytest.mark.integration
@pytest.mark.runpod
@pytest.mark.asyncio
async def test_runpod_offers_h100() -> None:
    from skyward.providers.runpod.config import RunPod

    provider = await RunPod().create_provider()
    offers = [o async for o in provider.offers(_gpu_spec("H100"))]
    _assert_offers(offers)


# ── Hyperstack ───────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.hyperstack
@pytest.mark.asyncio
async def test_hyperstack_offers_a100() -> None:
    from skyward.providers.hyperstack.config import Hyperstack

    provider = await Hyperstack().create_provider()
    offers = [o async for o in provider.offers(_gpu_spec("A100"))]
    _assert_offers(offers)


# ── TensorDock ───────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.tensordock
@pytest.mark.asyncio
async def test_tensordock_offers_a100() -> None:
    from skyward.providers.tensordock.config import TensorDock

    provider = await TensorDock().create_provider()
    offers = [o async for o in provider.offers(_gpu_spec("A100"))]
    _assert_offers(offers)


@pytest.mark.integration
@pytest.mark.tensordock
@pytest.mark.asyncio
async def test_tensordock_offers_h100() -> None:
    from skyward.providers.tensordock.config import TensorDock

    provider = await TensorDock().create_provider()
    offers = [o async for o in provider.offers(_gpu_spec("H100"))]
    _assert_offers(offers)


# ── Verda ────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.verda
@pytest.mark.asyncio
async def test_verda_offers_a100() -> None:
    from skyward.providers.verda.config import Verda

    provider = await Verda().create_provider()
    offers = [o async for o in provider.offers(_gpu_spec("A100"))]
    _assert_offers(offers)


@pytest.mark.integration
@pytest.mark.verda
@pytest.mark.asyncio
async def test_verda_offers_h100() -> None:
    from skyward.providers.verda.config import Verda

    provider = await Verda().create_provider()
    offers = [o async for o in provider.offers(_gpu_spec("H100"))]
    _assert_offers(offers)
