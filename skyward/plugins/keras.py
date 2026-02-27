"""Keras 3 plugin — environment + distributed training configuration."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal

from skyward.plugins.plugin import Plugin

if TYPE_CHECKING:
    from skyward.api.model import Cluster
    from skyward.api.runtime import InstanceInfo
    from skyward.api.spec import Image

type Backend = Literal["jax", "torch", "tensorflow"]


def keras(
    backend: Backend = "jax",
) -> Plugin:
    """Keras 3 plugin with backend configuration and distributed training.

    Pair with the matching backend plugin for multi-node training:

        plugins=[sky.plugins.jax(), sky.plugins.keras(backend="jax")]

    Parameters
    ----------
    backend
        Keras backend ("jax", "torch", or "tensorflow").
    """

    def transform(image: Image, cluster: Cluster[Any]) -> Image:
        return replace(
            image,
            pip=(*image.pip, "keras"),
            env={**image.env, "KERAS_BACKEND": backend},
        )

    @contextmanager
    def around(info: InstanceInfo) -> Iterator[None]:
        os.environ["KERAS_BACKEND"] = backend

        from skyward.observability.logger import logger

        log = logger.bind(plugin="keras", backend=backend)

        if info.total_nodes > 1 and backend == "jax":
            import keras as _keras  # type: ignore[reportMissingImports]

            devices = _keras.distribution.list_devices()
            log.debug("Available devices: {devices}", devices=devices)

            if devices:
                _keras.distribution.set_distribution(
                    _keras.distribution.DataParallel(
                        devices=devices,
                        auto_shard_dataset=False,
                    )
                )

                from keras.src.backend.jax.distribution_lib import (
                    initialize_rng,  # type: ignore[reportMissingImports]
                )
                initialize_rng()
                log.debug("Keras DataParallel distribution set, RNG synchronized")

        yield

    return (
        Plugin.create("keras")
        .with_image_transform(transform)
        .with_around_process(around)
    )
