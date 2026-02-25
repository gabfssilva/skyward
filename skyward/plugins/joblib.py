"""Joblib plugin — distributed parallel execution backend."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING

from skyward.plugins.plugin import Plugin

if TYPE_CHECKING:
    from skyward.api.pool import ComputePool
    from skyward.api.spec import Image


def joblib(version: str | None = None) -> Plugin:
    """Joblib plugin with Skyward parallel backend.

    Parameters
    ----------
    version
        Specific joblib version (e.g. "1.3.0"). None for latest.
    """

    def transform(image: Image) -> Image:
        pkg = f"joblib=={version}" if version else "joblib"
        return replace(image, pip=(*image.pip, pkg))

    @contextmanager
    def around_client(pool: ComputePool) -> Iterator[None]:
        from joblib import parallel_backend

        from skyward.integrations.joblib import _setup_backend, _strip_local_warning_filters

        _setup_backend(pool)
        _strip_local_warning_filters()
        with parallel_backend("skyward"):
            yield

    return (
        Plugin.create("joblib")
        .with_image_transform(transform)
        .with_around_client(around_client)
    )
