"""cuML plugin — RAPIDS GPU-accelerated ML with sklearn acceleration."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from skyward.api.spec import PipIndex
from skyward.plugins.plugin import Plugin

if TYPE_CHECKING:
    from skyward.api.model import Cluster
    from skyward.api.runtime import InstanceInfo
    from skyward.api.spec import Image


def cuml(cuda: str = "cu12") -> Plugin:
    """cuML plugin with NVIDIA index and sklearn acceleration.

    Parameters
    ----------
    cuda
        CUDA version suffix.
    """

    def transform(image: Image, cluster: Cluster[Any]) -> Image:
        return replace(
            image,
            pip=(*image.pip, f"cuml-{cuda}"),
            pip_indexes=(*image.pip_indexes, PipIndex(
                url="https://pypi.nvidia.com",
                packages=(f"cuml-{cuda}",),
            )),
        )

    @contextmanager
    def around(info: InstanceInfo) -> Iterator[None]:
        import cuml.accel  # type: ignore[reportMissingImports]

        cuml.accel.install()
        yield

    return (
        Plugin.create("cuml")
        .with_image_transform(transform)
        .with_around_process(around)
    )
