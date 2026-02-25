from __future__ import annotations

import pytest

from skyward.api.spec import Image

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class TestTorchPlugin:
    def test_factory_returns_plugin(self):
        from skyward.plugins.torch import torch
        p = torch()
        assert p.name == "torch"

    def test_transform_adds_pip_packages(self):
        from skyward.plugins.torch import torch
        p = torch()
        image = Image(python="3.13")
        assert p.transform is not None
        result = p.transform(image)
        assert "torch" in result.pip
        assert "torchvision" in result.pip
        assert "torchaudio" in result.pip

    def test_transform_adds_cuda_index(self):
        from skyward.plugins.torch import torch
        p = torch(cuda="cu124")
        image = Image(python="3.13")
        assert p.transform is not None
        result = p.transform(image)
        assert any("cu124" in idx.url for idx in result.pip_indexes)

    def test_custom_cuda_version(self):
        from skyward.plugins.torch import torch
        p = torch(cuda="cu118")
        image = Image(python="3.13")
        assert p.transform is not None
        result = p.transform(image)
        assert any("cu118" in idx.url for idx in result.pip_indexes)

    def test_transform_preserves_existing_pip(self):
        from skyward.plugins.torch import torch
        p = torch()
        image = Image(python="3.13", pip=["numpy", "pandas"])
        assert p.transform is not None
        result = p.transform(image)
        assert "numpy" in result.pip
        assert "pandas" in result.pip
        assert "torch" in result.pip

    def test_has_decorator(self):
        from skyward.plugins.torch import torch
        p = torch(backend="nccl")
        assert p.decorate is not None

    def test_no_around_app(self):
        from skyward.plugins.torch import torch
        p = torch()
        assert p.around_app is None

    def test_lazy_import(self):
        from skyward.plugins.torch import torch
        p = torch()
        assert p.name == "torch"
