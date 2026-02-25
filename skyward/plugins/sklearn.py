"""Scikit-learn plugin — distributed ML training backend."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from skyward.plugins.plugin import Plugin

if TYPE_CHECKING:
    from skyward.api.model import Cluster
    from skyward.api.pool import ComputePool
    from skyward.api.spec import Image


def sklearn(version: str | None = None) -> Plugin:
    """Scikit-learn plugin with Skyward parallel backend.

    Parameters
    ----------
    version
        Specific scikit-learn version (e.g. "1.4.0"). None for latest.
    """

    def transform(image: Image, cluster: Cluster[Any]) -> Image:
        pkg = f"scikit-learn=={version}" if version else "scikit-learn"
        return replace(image, pip=(*image.pip, pkg, "joblib"))

    @contextmanager
    def around_client(pool: ComputePool, cluster: Cluster[Any]) -> Iterator[None]:
        from joblib import parallel_backend

        from skyward.integrations.joblib import _setup_backend, _strip_local_warning_filters

        _setup_backend(pool)
        _strip_local_warning_filters()
        with parallel_backend("skyward"):
            yield

    return (
        Plugin.create("sklearn")
        .with_image_transform(transform)
        .with_around_client(around_client)
    )
