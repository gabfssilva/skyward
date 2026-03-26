from __future__ import annotations

import pytest

from unittest.mock import patch

from skyward.providers.runpod.config import RunPod
from skyward.providers.runpod.provider import RunPodProvider

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


def _fake_gpu_type(
    *,
    gpu_id: str = "NVIDIA A100 80GB PCIe",
    display_name: str = "A100 80GB",
    memory_gb: int = 80,
    spot_price: float | None = 0.50,
    on_demand_price: float | None = 1.00,
    min_vcpu: int = 4,
    min_memory: int = 16,
    max_gpu_count: int | None = None,
    max_gpu_count_secure: int | None = None,
    max_gpu_count_community: int | None = None,
) -> dict:
    gpu: dict = {
        "id": gpu_id,
        "displayName": display_name,
        "memoryInGb": memory_gb,
        "secureCloud": True,
        "communityCloud": True,
        "lowestPrice": {
            "minimumBidPrice": spot_price,
            "uninterruptablePrice": on_demand_price,
            "minVcpu": min_vcpu,
            "minMemory": min_memory,
        },
    }
    if max_gpu_count is not None:
        gpu["maxGpuCount"] = max_gpu_count
    if max_gpu_count_secure is not None:
        gpu["maxGpuCountSecureCloud"] = max_gpu_count_secure
    if max_gpu_count_community is not None:
        gpu["maxGpuCountCommunityCloud"] = max_gpu_count_community
    return gpu


async def _collect_offers(provider: RunPodProvider) -> list:
    return [offer async for offer in provider.offers()]


class TestMultiGpuOffers:
    @pytest.mark.asyncio
    async def test_offers_emit_multiple_gpu_counts(self) -> None:
        provider = RunPodProvider(RunPod())
        gpu = _fake_gpu_type(max_gpu_count_secure=4)

        with patch.object(provider, "_fetch_gpu_types", return_value=[(gpu, "secure", "12.4")]):
            offers = await _collect_offers(provider)

        counts = [o.instance_type.accelerator.count for o in offers]
        assert counts == [1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_offers_prices_scale_linearly(self) -> None:
        provider = RunPodProvider(RunPod())
        gpu = _fake_gpu_type(spot_price=0.50, on_demand_price=1.00, max_gpu_count_secure=3)

        with patch.object(provider, "_fetch_gpu_types", return_value=[(gpu, "secure", "12.4")]):
            offers = await _collect_offers(provider)

        assert len(offers) == 3
        assert offers[0].spot_price == 0.50
        assert offers[0].on_demand_price == 1.00
        assert offers[1].spot_price == 1.00
        assert offers[1].on_demand_price == 2.00
        assert offers[2].spot_price == 1.50
        assert offers[2].on_demand_price == 3.00

    @pytest.mark.asyncio
    async def test_offers_use_cloud_specific_max(self) -> None:
        provider = RunPodProvider(RunPod())
        gpu = _fake_gpu_type(
            max_gpu_count_secure=2,
            max_gpu_count_community=6,
            max_gpu_count=4,
        )

        with patch.object(
            provider, "_fetch_gpu_types",
            return_value=[(gpu, "secure", "12.4"), (gpu, "community", "12.4")],
        ):
            offers = await _collect_offers(provider)

        secure_counts = [
            o.instance_type.accelerator.count
            for o in offers if "secure" in o.id
        ]
        community_counts = [
            o.instance_type.accelerator.count
            for o in offers if "community" in o.id
        ]
        assert secure_counts == [1, 2]
        assert community_counts == [1, 2, 3, 4, 5, 6]

    @pytest.mark.asyncio
    async def test_offers_fallback_to_generic_max(self) -> None:
        provider = RunPodProvider(RunPod())
        gpu = _fake_gpu_type(max_gpu_count=3)

        with patch.object(provider, "_fetch_gpu_types", return_value=[(gpu, "secure", "12.4")]):
            offers = await _collect_offers(provider)

        counts = [o.instance_type.accelerator.count for o in offers]
        assert counts == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_offers_single_gpu_when_no_max(self) -> None:
        provider = RunPodProvider(RunPod())
        gpu = _fake_gpu_type()

        with patch.object(provider, "_fetch_gpu_types", return_value=[(gpu, "secure", "12.4")]):
            offers = await _collect_offers(provider)

        assert len(offers) == 1
        assert offers[0].instance_type.accelerator.count == 1
