from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from skyward.api.spec import Image

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class TestCuMLPlugin:
    def test_factory_returns_plugin(self):
        from skyward.plugins.cuml import cuml
        p = cuml()
        assert p.name == "cuml"

    def test_transform_adds_pip(self):
        from skyward.plugins.cuml import cuml
        p = cuml(cuda="cu12")
        image = Image(python="3.13")
        assert p.transform is not None
        result = p.transform(image, MagicMock())
        assert "cuml-cu12" in result.pip

    def test_transform_adds_nvidia_index(self):
        from skyward.plugins.cuml import cuml
        p = cuml()
        image = Image(python="3.13")
        assert p.transform is not None
        result = p.transform(image, MagicMock())
        assert any("pypi.nvidia.com" in idx.url for idx in result.pip_indexes)

    def test_has_around_process(self):
        from skyward.plugins.cuml import cuml
        p = cuml()
        assert p.around_process is not None

    def test_no_decorator(self):
        from skyward.plugins.cuml import cuml
        p = cuml()
        assert p.decorate is None

    def test_custom_cuda_version(self):
        from skyward.plugins.cuml import cuml
        p = cuml(cuda="cu11")
        image = Image(python="3.13")
        assert p.transform is not None
        result = p.transform(image, MagicMock())
        assert "cuml-cu11" in result.pip

    def test_lazy_import(self):
        from skyward.plugins.cuml import cuml
        p = cuml()
        assert p.name == "cuml"
