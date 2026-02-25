from __future__ import annotations

from dataclasses import dataclass

import pytest

from skyward.accelerators import Accelerator
from skyward.api.spec import Image

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


@dataclass(frozen=True)
class _FakeSpec:
    accelerator: Accelerator | str | None = None


@dataclass(frozen=True)
class _FakeCluster:
    spec: _FakeSpec = _FakeSpec()


def _gpu_cluster() -> _FakeCluster:
    return _FakeCluster(spec=_FakeSpec(
        accelerator=Accelerator.from_name("A100"),
    ))


def _cpu_cluster() -> _FakeCluster:
    return _FakeCluster(spec=_FakeSpec(accelerator=None))


class TestTorchPlugin:
    def test_factory_returns_plugin(self) -> None:
        from skyward.plugins.torch import torch
        p = torch()
        assert p.name == "torch"

    def test_transform_adds_pip_packages(self) -> None:
        from skyward.plugins.torch import torch
        p = torch(vision='latest', audio='latest')
        image = Image(python="3.13")
        assert p.transform is not None
        result = p.transform(image, _gpu_cluster())  # type: ignore[arg-type]
        assert "torch" in result.pip
        assert "torchvision" in result.pip
        assert "torchaudio" in result.pip

    def test_transform_adds_cuda_index(self) -> None:
        from skyward.plugins.torch import torch
        p = torch(cuda="cu124")
        image = Image(python="3.13")
        assert p.transform is not None
        result = p.transform(image, _gpu_cluster())  # type: ignore[arg-type]
        assert any("cu124" in idx.url for idx in result.pip_indexes)

    def test_transform_adds_cpu_index_without_gpu(self) -> None:
        from skyward.plugins.torch import torch
        p = torch()
        image = Image(python="3.13")
        assert p.transform is not None
        result = p.transform(image, _cpu_cluster())  # type: ignore[arg-type]
        assert any("cpu" in idx.url for idx in result.pip_indexes)

    def test_custom_cuda_version(self) -> None:
        from skyward.plugins.torch import torch
        p = torch(cuda="cu118")
        image = Image(python="3.13")
        assert p.transform is not None
        result = p.transform(image, _gpu_cluster())  # type: ignore[arg-type]
        assert any("cu118" in idx.url for idx in result.pip_indexes)

    def test_transform_preserves_existing_pip(self) -> None:
        from skyward.plugins.torch import torch
        p = torch()
        image = Image(python="3.13", pip=["numpy", "pandas"])
        assert p.transform is not None
        result = p.transform(image, _gpu_cluster())  # type: ignore[arg-type]
        assert "numpy" in result.pip
        assert "pandas" in result.pip
        assert "torch" in result.pip

    def test_has_decorator(self) -> None:
        from skyward.plugins.torch import torch
        p = torch(backend="nccl")
        assert p.decorate is not None

    def test_no_around_app(self) -> None:
        from skyward.plugins.torch import torch
        p = torch()
        assert p.around_app is None

    def test_lazy_import(self) -> None:
        from skyward.plugins.torch import torch
        p = torch()
        assert p.name == "torch"
