"""Keras 3 plugin — environment + distributed training configuration."""

from __future__ import annotations

import os
from dataclasses import replace
from functools import wraps
from typing import TYPE_CHECKING, Any, Literal

from skyward.plugins.plugin import Plugin

if TYPE_CHECKING:
    from collections.abc import Callable

    from skyward.api.model import Cluster
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

    def decorate[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            os.environ["KERAS_BACKEND"] = backend

            from skyward.api.runtime import instance_info
            from skyward.observability.logger import logger

            log = logger.bind(plugin="keras", backend=backend)
            info = instance_info()

            if info and info.total_nodes > 1 and backend == "jax":
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

            return fn(*args, **kwargs)

        return wrapper

    return (
        Plugin.create("keras")
        .with_image_transform(transform)
        .with_decorator(decorate)
    )
