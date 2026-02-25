from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from skyward.api.spec import Image

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class TestJAXPlugin:
    def test_factory_returns_plugin(self):
        from skyward.plugins.jax import jax
        p = jax()
        assert p.name == "jax"

    def test_transform_adds_pip(self):
        from skyward.plugins.jax import jax
        p = jax()
        image = Image(python="3.13")
        assert p.transform is not None
        result = p.transform(image, MagicMock())
        assert any("jax" in pkg for pkg in result.pip)

    def test_transform_adds_cuda_index(self):
        from skyward.plugins.jax import jax
        p = jax(cuda="cu124")
        image = Image(python="3.13")
        assert p.transform is not None
        result = p.transform(image, MagicMock())
        assert len(result.pip_indexes) > 0

    def test_has_around_app(self):
        from skyward.plugins.jax import jax
        p = jax()
        assert p.around_app is not None

    def test_no_decorator(self):
        from skyward.plugins.jax import jax
        p = jax()
        assert p.decorate is None

    def test_lazy_import(self):
        from skyward.plugins.jax import jax
        p = jax()
        assert p.name == "jax"
