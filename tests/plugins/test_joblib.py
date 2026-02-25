"""Tests for the joblib plugin."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from skyward.api.spec import Image

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class TestJoblibPlugin:
    def test_factory_returns_plugin(self) -> None:
        from skyward.plugins.joblib import joblib

        p = joblib()
        assert p.name == "joblib"

    def test_transform_adds_pip(self) -> None:
        from skyward.plugins.joblib import joblib

        p = joblib()
        image = Image(python="3.13")
        assert p.transform is not None
        result = p.transform(image, MagicMock())
        assert "joblib" in result.pip

    def test_versioned_pip(self) -> None:
        from skyward.plugins.joblib import joblib

        p = joblib(version="1.3.0")
        image = Image(python="3.13")
        assert p.transform is not None
        result = p.transform(image, MagicMock())
        assert "joblib==1.3.0" in result.pip

    def test_has_around_client(self) -> None:
        from skyward.plugins.joblib import joblib

        p = joblib()
        assert p.around_client is not None

    def test_no_decorator(self) -> None:
        from skyward.plugins.joblib import joblib

        p = joblib()
        assert p.decorate is None

    def test_no_around_app(self) -> None:
        from skyward.plugins.joblib import joblib

        p = joblib()
        assert p.around_app is None

    def test_transform_preserves_existing_pip(self) -> None:
        from skyward.plugins.joblib import joblib

        p = joblib()
        image = Image(python="3.13", pip=["numpy"])
        assert p.transform is not None
        result = p.transform(image, MagicMock())
        assert "numpy" in result.pip
        assert "joblib" in result.pip

    def test_no_bootstrap(self) -> None:
        from skyward.plugins.joblib import joblib

        p = joblib()
        assert p.bootstrap is None
