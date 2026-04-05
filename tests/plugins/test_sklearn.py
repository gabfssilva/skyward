"""Tests for the scikit-learn plugin."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from skyward.core.spec import Image

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class TestSklearnPlugin:
    def test_transform_adds_pip(self) -> None:
        from skyward.plugins.sklearn import sklearn

        p = sklearn()
        image = Image(python="3.13")
        assert p.transform is not None
        result = p.transform(image, MagicMock())
        assert "scikit-learn" in result.pip
        assert "joblib" in result.pip

    def test_versioned_pip(self) -> None:
        from skyward.plugins.sklearn import sklearn

        p = sklearn(version="1.4.0")
        image = Image(python="3.13")
        assert p.transform is not None
        result = p.transform(image, MagicMock())
        assert "scikit-learn==1.4.0" in result.pip

    def test_transform_preserves_existing_pip(self) -> None:
        from skyward.plugins.sklearn import sklearn

        p = sklearn()
        image = Image(python="3.13", pip=["numpy"])
        assert p.transform is not None
        result = p.transform(image, MagicMock())
        assert "numpy" in result.pip
        assert "scikit-learn" in result.pip
