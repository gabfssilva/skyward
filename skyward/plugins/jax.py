"""JAX plugin — environment + distributed initialization."""

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


def jax(cuda: str = "cu124") -> Plugin:
    """JAX plugin with CUDA support and distributed initialization.

    Parameters
    ----------
    cuda
        CUDA version suffix for JAX wheel index.
    """

    def transform(image: Image, cluster: Cluster[Any]) -> Image:
        return replace(
            image,
            pip=(*image.pip, f"jax[{cuda}]"),
            pip_indexes=(*image.pip_indexes, PipIndex(
                url="https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
                packages=("jax", "jaxlib"),
            )),
        )

    @contextmanager
    def around(info: InstanceInfo) -> Iterator[None]:
        import jax as _jax  # type: ignore[reportMissingImports]

        _jax.distributed.initialize(
            coordinator_address=f"{info.head_addr}:{info.head_port}",
            num_processes=info.total_nodes,
            process_id=info.node,
        )
        yield

    return (
        Plugin.create("jax")
        .with_image_transform(transform)
        .with_around_app(around)
    )
