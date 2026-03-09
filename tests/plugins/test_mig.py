"""Tests for the NVIDIA MIG plugin."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from skyward.core.spec import Image
from skyward.observability.metrics import GPU, Default
from skyward.plugins.mig import (
    _GPU_METRIC_NAMES,
    _SMI_MIG_MEM_TOTAL,
    _SMI_MIG_MEM_USED,
    _dcgm_query,
    mig,
)
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

    def test_transform_replaces_gpu_metrics_with_mig_aware(self) -> None:
        p = mig(profile="3g.40gb")
        image = Image(python="3.13", metrics=Default())
        assert p.transform is not None
        result = p.transform(image, MagicMock())
        assert result.metrics is not None
        metric_names = {m.name for m in result.metrics}
        assert "gpu_util" in metric_names
        assert "gpu_mem_mb" in metric_names
        assert "gpu_mem_total_mb" in metric_names
        assert "gpu_temp" in metric_names
        for m in result.metrics:
            if m.name in _GPU_METRIC_NAMES:
                assert m.multi is True

    def test_transform_metrics_use_dcgm_with_fallback(self) -> None:
        p = mig(profile="3g.40gb")
        image = Image(python="3.13", metrics=Default())
        assert p.transform is not None
        result = p.transform(image, MagicMock())
        assert result.metrics is not None
        gpu_mem = next(m for m in result.metrics if m.name == "gpu_mem_mb")
        assert "dcgmi" in gpu_mem.command
        assert "nvidia-smi" in gpu_mem.command

    def test_transform_gpu_temp_uses_nvidia_smi_directly(self) -> None:
        p = mig(profile="3g.40gb")
        image = Image(python="3.13", metrics=Default())
        assert p.transform is not None
        result = p.transform(image, MagicMock())
        assert result.metrics is not None
        temp = next(m for m in result.metrics if m.name == "gpu_temp")
        assert "nvidia-smi" in temp.command
        assert "temperature.gpu" in temp.command
        assert "dcgmi" not in temp.command

    def test_transform_preserves_non_gpu_metrics(self) -> None:
        p = mig(profile="3g.40gb")
        image = Image(python="3.13", metrics=Default())
        assert p.transform is not None
        result = p.transform(image, MagicMock())
        assert result.metrics is not None
        metric_names = {m.name for m in result.metrics}
        assert "cpu" in metric_names
        assert "mem" in metric_names

    def test_transform_preserves_gpu_interval(self) -> None:
        p = mig(profile="3g.40gb")
        image = Image(python="3.13", metrics=(GPU(interval=5),))
        assert p.transform is not None
        result = p.transform(image, MagicMock())
        assert result.metrics is not None
        gpu_util = next(m for m in result.metrics if m.name == "gpu_util")
        assert gpu_util.interval == 5

    def test_transform_handles_no_metrics(self) -> None:
        p = mig(profile="3g.40gb")
        image = Image(python="3.13", metrics=None)
        assert p.transform is not None
        result = p.transform(image, MagicMock())
        assert result.metrics is not None
        assert len(result.metrics) == 4  # the 4 MIG GPU metrics


class TestDCGMQuery:
    def test_dcgm_query_uses_dcgmi_dmon(self) -> None:
        cmd = _dcgm_query(252)
        assert "dcgmi dmon" in cmd
        assert "-e 252" in cmd

    def test_dcgm_query_discovers_entity_ids(self) -> None:
        cmd = _dcgm_query(1002)
        assert "dcgmi discovery -c" in cmd
        assert "-i $ids" in cmd

    def test_dcgm_query_parses_gpu_instance_rows(self) -> None:
        cmd = _dcgm_query(250)
        assert "GPU-I" in cmd

    def test_dcgm_query_one_shot(self) -> None:
        cmd = _dcgm_query(1002)
        assert "-c 1" in cmd

    def test_dcgm_query_filters_numeric_only(self) -> None:
        cmd = _dcgm_query(252)
        assert "grep -E" in cmd


class TestNvidiaSMIFallback:
    def test_smi_mem_used_parses_mig_table(self) -> None:
        assert "MIG devices" in _SMI_MIG_MEM_USED
        assert "NR%2==1" in _SMI_MIG_MEM_USED

    def test_smi_mem_total_parses_mig_table(self) -> None:
        assert "MIG devices" in _SMI_MIG_MEM_TOTAL
        assert "NR%2==1" in _SMI_MIG_MEM_TOTAL


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

    def test_bootstrap_installs_dcgm(self) -> None:
        p = mig(profile="3g.40gb")
        assert p.bootstrap is not None
        ops = p.bootstrap(self._mock_cluster())
        script = "\n".join(resolve(op) for op in ops)
        assert "datacenter-gpu-manager" in script
        assert "nv-hostengine" in script


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
