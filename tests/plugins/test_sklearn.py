"""Tests for the scikit-learn plugin."""

from __future__ import annotations

import pytest

from skyward.api.spec import Image

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class TestSklearnPlugin:
    def test_factory_returns_plugin(self) -> None:
        from skyward.plugins.sklearn import sklearn

        p = sklearn()
        assert p.name == "sklearn"

    def test_transform_adds_pip(self) -> None:
        from skyward.plugins.sklearn import sklearn

        p = sklearn()
        image = Image(python="3.13")
        assert p.transform is not None
        result = p.transform(image)
        assert "scikit-learn" in result.pip
        assert "joblib" in result.pip

    def test_versioned_pip(self) -> None:
        from skyward.plugins.sklearn import sklearn

        p = sklearn(version="1.4.0")
        image = Image(python="3.13")
        assert p.transform is not None
        result = p.transform(image)
        assert "scikit-learn==1.4.0" in result.pip

    def test_has_around_client(self) -> None:
        from skyward.plugins.sklearn import sklearn

        p = sklearn()
        assert p.around_client is not None

    def test_no_decorator(self) -> None:
        from skyward.plugins.sklearn import sklearn

        p = sklearn()
        assert p.decorate is None

    def test_no_around_app(self) -> None:
        from skyward.plugins.sklearn import sklearn

        p = sklearn()
        assert p.around_app is None

    def test_transform_preserves_existing_pip(self) -> None:
        from skyward.plugins.sklearn import sklearn

        p = sklearn()
        image = Image(python="3.13", pip=["numpy"])
        assert p.transform is not None
        result = p.transform(image)
        assert "numpy" in result.pip
        assert "scikit-learn" in result.pip

    def test_no_bootstrap(self) -> None:
        from skyward.plugins.sklearn import sklearn

        p = sklearn()
        assert p.bootstrap == ()
