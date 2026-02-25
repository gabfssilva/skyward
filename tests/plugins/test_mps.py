"""Tests for the NVIDIA MPS plugin."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from skyward.api.spec import Image
from skyward.plugins.mps import mps
from skyward.providers.bootstrap import resolve

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class TestMPSPlugin:
    def test_transform_adds_mps_env_vars(self) -> None:
        p = mps()
        image = Image(python="3.13")
        assert p.transform is not None
        result = p.transform(image, MagicMock())
        assert result.env["CUDA_MPS_PIPE_DIRECTORY"] == "/tmp/nvidia-mps"
        assert result.env["CUDA_MPS_LOG_DIRECTORY"] == "/tmp/nvidia-mps-log"

    def test_bootstrap_starts_mps_daemon(self) -> None:
        p = mps()
        assert p.bootstrap is not None
        ops = p.bootstrap(MagicMock())
        script = "\n".join(resolve(op) for op in ops)
        assert "nvidia-cuda-mps-control -d" in script

    def test_active_thread_percentage_sets_env(self) -> None:
        p = mps(active_thread_percentage=25)
        image = Image(python="3.13")
        assert p.transform is not None
        result = p.transform(image, MagicMock())
        assert result.env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] == "25"

    def test_pinned_memory_limit_sets_env(self) -> None:
        p = mps(pinned_memory_limit="0=2G")
        image = Image(python="3.13")
        assert p.transform is not None
        result = p.transform(image, MagicMock())
        assert result.env["CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"] == "0=2G"
