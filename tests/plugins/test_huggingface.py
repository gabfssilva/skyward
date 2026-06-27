"""Tests for the Hugging Face pre-download plugin."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from skyward.core.spec import Image
from skyward.plugins.huggingface import huggingface
from skyward.providers.bootstrap import resolve

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class TestHuggingFaceTransform:
    def test_transform_adds_huggingface_hub(self) -> None:
        p = huggingface(models=["org/model"])
        assert p.transform is not None
        result = p.transform(Image(python="3.13"), MagicMock())
        assert "huggingface_hub" in result.pip

    def test_transform_enables_xet_high_performance(self) -> None:
        p = huggingface(models=["org/model"])
        assert p.transform is not None
        result = p.transform(Image(python="3.13"), MagicMock())
        assert result.env["HF_XET_HIGH_PERFORMANCE"] == "1"

    def test_transform_sets_token_when_provided(self) -> None:
        p = huggingface(models=["org/model"], token="hf_secret")
        assert p.transform is not None
        result = p.transform(Image(python="3.13"), MagicMock())
        assert result.env["HF_TOKEN"] == "hf_secret"

    def test_transform_omits_token_when_absent(self) -> None:
        p = huggingface(models=["org/model"])
        assert p.transform is not None
        result = p.transform(Image(python="3.13"), MagicMock())
        assert "HF_TOKEN" not in result.env

    def test_transform_preserves_existing_env_and_pip(self) -> None:
        p = huggingface(models=["org/model"])
        assert p.transform is not None
        image = Image(python="3.13", pip=("numpy",), env={"EXISTING": "value"})
        result = p.transform(image, MagicMock())
        assert "numpy" in result.pip
        assert result.env["EXISTING"] == "value"


class TestHuggingFaceBootstrap:
    def test_bootstrap_downloads_each_model(self) -> None:
        p = huggingface(models=["org/a", "org/b"])
        assert p.bootstrap is not None
        script = "\n".join(resolve(op) for op in p.bootstrap(MagicMock()))
        assert "snapshot_download" in script
        assert "cached" in script
        assert '"org/a"' in script
        assert '"org/b"' in script

    def test_bootstrap_uses_venv_python(self) -> None:
        p = huggingface(models=["org/a"])
        assert p.bootstrap is not None
        script = "\n".join(resolve(op) for op in p.bootstrap(MagicMock()))
        assert "/opt/skyward/.venv/bin/python" in script
