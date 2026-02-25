"""Tests for the Hugging Face plugin."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from skyward.api.spec import Image

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class TestHuggingFacePlugin:
    def test_factory_returns_plugin(self) -> None:
        from skyward.plugins.huggingface import huggingface

        p = huggingface(token="hf_xxx")
        assert p.name == "huggingface"

    def test_transform_adds_pip(self) -> None:
        from skyward.plugins.huggingface import huggingface

        p = huggingface(token="hf_xxx")
        image = Image(python="3.13")
        assert p.transform is not None
        result = p.transform(image, MagicMock())
        assert "transformers" in result.pip
        assert "datasets" in result.pip
        assert "tokenizers" in result.pip

    def test_transform_adds_hf_token_env(self) -> None:
        from skyward.plugins.huggingface import huggingface

        p = huggingface(token="hf_test123")
        image = Image(python="3.13")
        assert p.transform is not None
        result = p.transform(image, MagicMock())
        assert result.env["HF_TOKEN"] == "hf_test123"

    def test_has_bootstrap_ops(self) -> None:
        from skyward.plugins.huggingface import huggingface

        p = huggingface(token="hf_xxx")
        assert p.bootstrap is not None
        result = p.bootstrap(MagicMock())
        assert len(result) > 0

    def test_no_decorator(self) -> None:
        from skyward.plugins.huggingface import huggingface

        p = huggingface(token="hf_xxx")
        assert p.decorate is None

    def test_transform_preserves_existing_env(self) -> None:
        from skyward.plugins.huggingface import huggingface

        p = huggingface(token="hf_xxx")
        image = Image(python="3.13", env={"EXISTING": "val"})
        assert p.transform is not None
        result = p.transform(image, MagicMock())
        assert result.env["EXISTING"] == "val"
        assert result.env["HF_TOKEN"] == "hf_xxx"

    def test_transform_preserves_existing_pip(self) -> None:
        from skyward.plugins.huggingface import huggingface

        p = huggingface(token="hf_xxx")
        image = Image(python="3.13", pip=["numpy"])
        assert p.transform is not None
        result = p.transform(image, MagicMock())
        assert "numpy" in result.pip
        assert "transformers" in result.pip

    def test_no_around_app(self) -> None:
        from skyward.plugins.huggingface import huggingface

        p = huggingface(token="hf_xxx")
        assert p.around_app is None

    def test_no_around_client(self) -> None:
        from skyward.plugins.huggingface import huggingface

        p = huggingface(token="hf_xxx")
        assert p.around_client is None
