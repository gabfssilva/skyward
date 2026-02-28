"""Tests for the NVIDIA MIG plugin."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from skyward.api.spec import Image
from skyward.plugins.mig import mig
from skyward.providers.bootstrap import resolve

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class TestMIGTransform:
    def test_transform_sets_nvidia_visible_devices(self) -> None:
        p = mig(profile="3g.40gb")
        image = Image(python="3.13")
        assert p.transform is not None
        result = p.transform(image, MagicMock())
        assert result.env["NVIDIA_VISIBLE_DEVICES"] == "all"

    def test_transform_preserves_existing_env(self) -> None:
        p = mig(profile="3g.40gb")
        image = Image(python="3.13", env={"EXISTING": "value"})
        assert p.transform is not None
        result = p.transform(image, MagicMock())
        assert result.env["EXISTING"] == "value"
        assert result.env["NVIDIA_VISIBLE_DEVICES"] == "all"


class TestMIGBootstrap:
    def _mock_cluster(self, concurrency: int = 2) -> MagicMock:
        cluster = MagicMock()
        cluster.spec.worker.concurrency = concurrency
        return cluster

    def test_bootstrap_enables_mig_mode(self) -> None:
        p = mig(profile="3g.40gb")
        assert p.bootstrap is not None
        ops = p.bootstrap(self._mock_cluster())
        script = "\n".join(resolve(op) for op in ops)
        assert "nvidia-smi -mig 1" in script

    def test_bootstrap_creates_partitions_matching_concurrency(self) -> None:
        p = mig(profile="3g.40gb")
        assert p.bootstrap is not None
        ops = p.bootstrap(self._mock_cluster(concurrency=2))
        script = "\n".join(resolve(op) for op in ops)
        assert script.count("nvidia-smi mig -cgi 3g.40gb -C") == 2

    def test_bootstrap_uses_profile_in_cgi_command(self) -> None:
        p = mig(profile="1g.10gb")
        assert p.bootstrap is not None
        ops = p.bootstrap(self._mock_cluster(concurrency=7))
        script = "\n".join(resolve(op) for op in ops)
        assert script.count("nvidia-smi mig -cgi 1g.10gb -C") == 7


class TestMIGRegistration:
    def test_mig_in_plugins_all(self) -> None:
        from skyward import plugins
        assert "mig" in plugins.__all__

    def test_mig_in_lazy_imports(self) -> None:
        from skyward.plugins import _LAZY_IMPORTS
        assert "mig" in _LAZY_IMPORTS
        assert _LAZY_IMPORTS["mig"] == ("skyward.plugins.mig", "mig")


class TestMIGAroundProcess:
    def _make_info(self, worker: int = 0) -> MagicMock:
        info = MagicMock()
        info.worker = worker
        return info

    def test_around_process_sets_cuda_visible_devices(self) -> None:
        p = mig(profile="3g.40gb")
        assert p.around_process is not None

        nvidia_smi_output = (
            "GPU 0: NVIDIA A100-SXM4-80GB (UUID: GPU-xxxx)\n"
            "  MIG 3g.40gb  Device  0: (UUID: MIG-aaaa)\n"
            "  MIG 3g.40gb  Device  1: (UUID: MIG-bbbb)\n"
        )
        with (
            patch("subprocess.run") as mock_run,
            patch.dict("os.environ", {}, clear=False),
        ):
            mock_run.return_value = MagicMock(stdout=nvidia_smi_output, returncode=0)
            with p.around_process(self._make_info(worker=0)):
                import os
                assert os.environ["CUDA_VISIBLE_DEVICES"] == "MIG-aaaa"

    def test_around_process_assigns_correct_partition_by_worker_index(self) -> None:
        p = mig(profile="3g.40gb")
        assert p.around_process is not None

        nvidia_smi_output = (
            "GPU 0: NVIDIA A100-SXM4-80GB (UUID: GPU-xxxx)\n"
            "  MIG 3g.40gb  Device  0: (UUID: MIG-aaaa)\n"
            "  MIG 3g.40gb  Device  1: (UUID: MIG-bbbb)\n"
        )
        with (
            patch("subprocess.run") as mock_run,
            patch.dict("os.environ", {}, clear=False),
        ):
            mock_run.return_value = MagicMock(stdout=nvidia_smi_output, returncode=0)
            with p.around_process(self._make_info(worker=1)):
                import os
                assert os.environ["CUDA_VISIBLE_DEVICES"] == "MIG-bbbb"
