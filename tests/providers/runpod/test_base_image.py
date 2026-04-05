from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from skyward.accelerators import Accelerator
from skyward.core import Nodes, PoolSpec
from skyward.providers.runpod.config import RunPod
from skyward.providers.runpod.provider import (
    _fetch_docker_tags,
    _resolve_image_candidates,
    _select_image_candidates,
)

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


def _spec_with_cuda(cuda_min: str = "12.1", cuda_max: str = "12.9") -> PoolSpec:
    return PoolSpec(
        nodes=Nodes(desired=1),
        region="global",
        accelerator=Accelerator(
            name="A100",
            memory="80GB",
            metadata={"cuda": {"min": cuda_min, "max": cuda_max}},
        ),
    )


class TestFetchDockerTags:
    @pytest.mark.asyncio
    async def test_fetches_from_correct_repo(self) -> None:
        with patch("skyward.providers.runpod.provider.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"results": [{"name": "tag1"}], "next": None}
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            tags = await _fetch_docker_tags("nvidia/cuda")

            call_args = mock_client.get.call_args
            assert "/nvidia/cuda/" in call_args[0][0]
            assert tags == ["tag1"]


class TestSelectImageCandidates:
    def test_uses_repo_in_output(self) -> None:
        tags = ["12.4.0-cudnn-runtime-ubuntu22.04"]
        result = _select_image_candidates(
            tags, cuda_min=(12, 0), cuda_max=(12, 9), ubuntu="newest", repo="nvidia/cuda",
        )
        assert all("nvidia/cuda:" in img for img in result)

    def test_defaults_to_nvidia_repo(self) -> None:
        tags = ["12.4.0-cudnn-runtime-ubuntu22.04"]
        result = _select_image_candidates(
            tags, cuda_min=(12, 0), cuda_max=(12, 9), ubuntu="newest",
        )
        assert all("nvidia/cuda:" in img for img in result)

    def test_filters_by_variant(self) -> None:
        tags = [
            "12.8.0-cudnn-runtime-ubuntu24.04",
            "12.8.0-runtime-ubuntu24.04",
            "12.8.0-devel-ubuntu24.04",
            "12.8.0-base-ubuntu24.04",
        ]
        result = _select_image_candidates(
            tags, cuda_min=(12, 0), cuda_max=(12, 9), ubuntu="newest",
        )
        assert len(result) == 1
        assert "cudnn-runtime" in result[0]

    def test_matches_cudnn_versioned_variant(self) -> None:
        tags = [
            "12.4.0-cudnn9-runtime-ubuntu24.04",
            "12.4.0-runtime-ubuntu24.04",
            "12.4.0-devel-ubuntu24.04",
        ]
        result = _select_image_candidates(
            tags, cuda_min=(12, 0), cuda_max=(12, 9), ubuntu="newest",
        )
        assert len(result) == 1
        assert "cudnn9-runtime" in result[0]

    def test_selects_by_cuda_range(self) -> None:
        tags = [
            "13.2.0-cudnn-runtime-ubuntu24.04",
            "12.9.1-cudnn-runtime-ubuntu24.04",
            "12.8.0-cudnn-runtime-ubuntu24.04",
            "12.4.0-cudnn-runtime-ubuntu24.04",
            "11.8.0-cudnn-runtime-ubuntu24.04",
        ]
        result = _select_image_candidates(
            tags, cuda_min=(12, 4), cuda_max=(13, 1), ubuntu="newest",
        )
        assert len(result) == 3
        assert "12.9.1" in result[0]
        assert "12.8.0" in result[1]
        assert "12.4.0" in result[2]

    def test_picks_highest_patch_per_cuda_minor(self) -> None:
        tags = [
            "12.8.1-cudnn-runtime-ubuntu24.04",
            "12.8.0-cudnn-runtime-ubuntu24.04",
        ]
        result = _select_image_candidates(
            tags, cuda_min=(12, 0), cuda_max=(12, 9), ubuntu="newest",
        )
        assert len(result) == 1
        assert "12.8.1" in result[0]


class TestResolveImageCandidates:
    @pytest.mark.asyncio
    async def test_container_image_overrides(self) -> None:
        config = RunPod(container_image="custom/image:latest")
        result = await _resolve_image_candidates(_spec_with_cuda(), config)
        assert result == ("custom/image:latest",)

    @pytest.mark.asyncio
    async def test_resolves_nvidia_cuda_images(self) -> None:
        config = RunPod(base_image="nvidia")
        nvidia_tags = [
            "12.8.0-cudnn-runtime-ubuntu24.04",
            "12.4.0-cudnn-runtime-ubuntu22.04",
        ]
        with patch(
            "skyward.providers.runpod.provider._fetch_docker_tags",
            return_value=nvidia_tags,
        ) as mock_fetch:
            result = await _resolve_image_candidates(_spec_with_cuda(), config)

        mock_fetch.assert_called_once_with("nvidia/cuda")
        assert all("nvidia/cuda:" in img for img in result)

    @pytest.mark.asyncio
    async def test_no_candidates_uses_fallback(self) -> None:
        config = RunPod()
        with patch(
            "skyward.providers.runpod.provider._fetch_docker_tags",
            return_value=[],
        ):
            result = await _resolve_image_candidates(_spec_with_cuda(), config)
        assert "runpod/base:" in result[0]

    @pytest.mark.asyncio
    async def test_no_cuda_range_uses_fallback(self) -> None:
        config = RunPod()
        spec = PoolSpec(nodes=Nodes(desired=1), region="global", accelerator=None)
        result = await _resolve_image_candidates(spec, config)
        assert "runpod/base:" in result[0]

    @pytest.mark.asyncio
    async def test_ubuntu_filter(self) -> None:
        config = RunPod(base_image="nvidia", ubuntu="22.04")
        tags = [
            "12.8.0-cudnn-runtime-ubuntu24.04",
            "12.8.0-cudnn-runtime-ubuntu22.04",
        ]
        with patch(
            "skyward.providers.runpod.provider._fetch_docker_tags",
            return_value=tags,
        ):
            result = await _resolve_image_candidates(_spec_with_cuda(), config)
        assert len(result) == 1
        assert "ubuntu22.04" in result[0]
