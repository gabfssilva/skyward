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
        nodes=Nodes(min=1),
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
        """_fetch_docker_tags should use the repo parameter, not a hardcoded constant."""
        with patch("skyward.providers.runpod.provider.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"results": [{"name": "tag1"}], "next": None}
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            tags = await _fetch_docker_tags("runpod/pytorch")

            call_args = mock_client.get.call_args
            assert "/runpod/pytorch/" in call_args[0][0]
            assert tags == ["tag1"]


class TestSelectImageCandidates:
    def test_uses_repo_in_output(self) -> None:
        """_select_image_candidates should prefix image names with the given repo."""
        tags = ["1.0.3-cuda1240-ubuntu2204"]
        result = _select_image_candidates(
            tags, cuda_min=(12, 0), cuda_max=(12, 9), ubuntu="newest", repo="runpod/pytorch",
        )
        assert all("runpod/pytorch:" in img for img in result)

    def test_defaults_to_base_repo(self) -> None:
        """_select_image_candidates should default repo to runpod/base."""
        tags = ["1.0.3-cuda1240-ubuntu2204"]
        result = _select_image_candidates(
            tags, cuda_min=(12, 0), cuda_max=(12, 9), ubuntu="newest",
        )
        assert all("runpod/base:" in img for img in result)


class TestResolveImageCandidates:
    @pytest.mark.asyncio
    async def test_container_image_overrides_base_image(self) -> None:
        config = RunPod(base_image="pytorch", container_image="custom/image:latest")
        result = await _resolve_image_candidates(_spec_with_cuda(), config)
        assert result == ("custom/image:latest",)

    @pytest.mark.asyncio
    async def test_base_image_pytorch_uses_pytorch_repo(self) -> None:
        config = RunPod(base_image="pytorch")
        pytorch_tags = [
            "2.8.0-py3.13-cuda12.8.1-devel-ubuntu24.04",
            "2.7.0-py3.12-cuda12.4.0-devel-ubuntu22.04",
        ]
        with patch(
            "skyward.providers.runpod.provider._fetch_docker_tags",
            return_value=pytorch_tags,
        ) as mock_fetch:
            result = await _resolve_image_candidates(_spec_with_cuda(), config)

        mock_fetch.assert_called_once_with("runpod/pytorch")
        assert all("runpod/pytorch:" in img for img in result)

    @pytest.mark.asyncio
    async def test_base_image_base_uses_base_repo(self) -> None:
        config = RunPod(base_image="base")
        base_tags = ["1.0.3-cuda1240-ubuntu2204"]
        with patch(
            "skyward.providers.runpod.provider._fetch_docker_tags",
            return_value=base_tags,
        ) as mock_fetch:
            result = await _resolve_image_candidates(_spec_with_cuda(), config)

        mock_fetch.assert_called_once_with("runpod/base")
        assert all("runpod/base:" in img for img in result)

    @pytest.mark.asyncio
    async def test_pytorch_no_candidates_raises(self) -> None:
        """When base_image='pytorch' and no tags match, raise instead of silent fallback."""
        config = RunPod(base_image="pytorch")
        with (
            patch("skyward.providers.runpod.provider._fetch_docker_tags", return_value=[]),
            pytest.raises(RuntimeError, match="No matching.*pytorch"),
        ):
            await _resolve_image_candidates(_spec_with_cuda(), config)

    @pytest.mark.asyncio
    async def test_base_no_candidates_uses_fallback(self) -> None:
        """When base_image='base' and no tags match, use hardcoded fallback (existing behavior)."""
        config = RunPod(base_image="base")
        with patch(
            "skyward.providers.runpod.provider._fetch_docker_tags",
            return_value=[],
        ):
            result = await _resolve_image_candidates(_spec_with_cuda(), config)
        assert "runpod/base:" in result[0]

    @pytest.mark.asyncio
    async def test_pytorch_no_cuda_range_raises(self) -> None:
        """When base_image='pytorch' and no CUDA range, raise instead of silent fallback."""
        config = RunPod(base_image="pytorch")
        spec = PoolSpec(nodes=Nodes(min=1), region="global", accelerator=None)
        with pytest.raises(RuntimeError, match="No CUDA range.*pytorch"):
            await _resolve_image_candidates(spec, config)

    @pytest.mark.asyncio
    async def test_base_no_cuda_range_uses_fallback(self) -> None:
        """When base_image='base' and no CUDA range, use fallback (existing behavior)."""
        config = RunPod(base_image="base")
        spec = PoolSpec(nodes=Nodes(min=1), region="global", accelerator=None)
        result = await _resolve_image_candidates(spec, config)
        assert "runpod/base:" in result[0]


class TestPytorchTagParsing:
    """Verify _select_image_candidates handles pytorch tag format correctly."""

    def test_selects_pytorch_tags_by_cuda_range(self) -> None:
        tags = [
            "2.8.0-py3.13-cuda12.8.1-devel-ubuntu24.04",
            "2.7.0-py3.12-cuda12.4.0-devel-ubuntu22.04",
            "2.6.0-py3.12-cuda12.1.0-devel-ubuntu22.04",
            "2.5.0-py3.11-cuda11.8.0-devel-ubuntu22.04",
        ]
        result = _select_image_candidates(
            tags,
            cuda_min=(12, 1),
            cuda_max=(12, 9),
            ubuntu="newest",
            repo="runpod/pytorch",
        )
        assert len(result) == 3
        assert all("runpod/pytorch:" in r for r in result)
        assert "cuda12.8" in result[0]
        assert "cuda12.4" in result[1]
        assert "cuda12.1" in result[2]

    def test_picks_newest_pytorch_version_per_cuda(self) -> None:
        tags = [
            "2.8.0-py3.13-cuda12.8.1-devel-ubuntu24.04",
            "2.7.0-py3.13-cuda12.8.1-devel-ubuntu24.04",
        ]
        result = _select_image_candidates(
            tags,
            cuda_min=(12, 1),
            cuda_max=(12, 9),
            ubuntu="newest",
            repo="runpod/pytorch",
        )
        assert len(result) == 1
        assert "2.8.0" in result[0]

    def test_ubuntu_filter_works_with_pytorch_format(self) -> None:
        tags = [
            "2.8.0-py3.13-cuda12.8.1-devel-ubuntu24.04",
            "2.8.0-py3.13-cuda12.8.1-devel-ubuntu22.04",
        ]
        result = _select_image_candidates(
            tags,
            cuda_min=(12, 1),
            cuda_max=(12, 9),
            ubuntu="22.04",
            repo="runpod/pytorch",
        )
        assert len(result) == 1
        assert "ubuntu22.04" in result[0]
