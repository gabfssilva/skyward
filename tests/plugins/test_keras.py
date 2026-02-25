"""Tests for the Keras plugin."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from skyward.api.spec import Image

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class TestKerasPlugin:
    def test_factory_returns_plugin(self) -> None:
        from skyward.plugins.keras import keras

        p = keras()
        assert p.name == "keras"

    def test_transform_adds_pip(self) -> None:
        from skyward.plugins.keras import keras

        p = keras()
        image = Image(python="3.13")
        assert p.transform is not None
        result = p.transform(image, MagicMock())
        assert "keras" in result.pip

    def test_transform_sets_backend_env(self) -> None:
        from skyward.plugins.keras import keras

        p = keras(backend="jax")
        image = Image(python="3.13")
        assert p.transform is not None
        result = p.transform(image, MagicMock())
        assert result.env["KERAS_BACKEND"] == "jax"

    def test_transform_torch_backend(self) -> None:
        from skyward.plugins.keras import keras

        p = keras(backend="torch")
        image = Image(python="3.13")
        assert p.transform is not None
        result = p.transform(image, MagicMock())
        assert result.env["KERAS_BACKEND"] == "torch"

    def test_has_decorator(self) -> None:
        from skyward.plugins.keras import keras

        p = keras()
        assert p.decorate is not None

    def test_no_around_app(self) -> None:
        from skyward.plugins.keras import keras

        p = keras()
        assert p.around_app is None

    def test_no_around_client(self) -> None:
        from skyward.plugins.keras import keras

        p = keras()
        assert p.around_client is None

    def test_transform_preserves_existing(self) -> None:
        from skyward.plugins.keras import keras

        p = keras()
        image = Image(python="3.13", pip=["numpy"], env={"KEY": "val"})
        assert p.transform is not None
        result = p.transform(image, MagicMock())
        assert "numpy" in result.pip
        assert "keras" in result.pip
        assert result.env["KEY"] == "val"

    def test_default_backend_is_jax(self) -> None:
        from skyward.plugins.keras import keras

        p = keras()
        image = Image(python="3.13")
        assert p.transform is not None
        result = p.transform(image, MagicMock())
        assert result.env["KERAS_BACKEND"] == "jax"
