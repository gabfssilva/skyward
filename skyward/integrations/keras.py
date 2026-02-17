"""Keras 3 distributed training integration."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Literal

Backend = Literal["jax", "torch", "tensorflow"] | None


def keras[**P, R](
    backend: Backend = None,
    seed: int | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Configure Keras 3 distributed training.

    Args:
        backend: Backend to use (jax, torch, tensorflow). Defaults to jax.
        seed: Random seed for reproducibility. Set before distribution init
            to ensure consistent RNG state across all processes.

    Example:
        >>> import skyward as sky

        >>> @sky.compute
        ... @sky.integrations.keras(backend="jax", seed=42)
        ... def train():
        ...     import keras
        ...     model = keras.Sequential([...])
        ...     model.fit(...)
    """

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        effective = backend or "jax"

        @functools.wraps(fn)
        def inner(*args: P.args, **kwargs: P.kwargs) -> R:
            from skyward import instance_info
            from skyward.observability.logger import logger

            log = logger.bind(integration="keras", backend=effective)
            pool = instance_info()

            log.debug("Using backend: {backend}", backend=effective)
            log.debug("Pool info: {pool}", pool=pool)

            if pool and pool.total_nodes > 1:
                import keras

                devices = keras.distribution.list_devices()
                log.debug("Available devices: {devices}", devices=devices)

                if devices:
                    log.debug("Setting up DataParallel distribution")
                    keras.distribution.set_distribution(
                        keras.distribution.DataParallel(
                            devices=devices,
                            auto_shard_dataset=False,
                        )
                    )
                    log.debug("DataParallel distribution set")

                    if effective == "jax":
                        from keras.src.backend.jax.distribution_lib import initialize_rng
                        initialize_rng()
                        log.debug("RNG synchronized across processes")

                if seed is not None:
                    keras.utils.set_random_seed(seed)
                    log.debug("Random seed set to {seed}", seed=seed)

            log.info("Keras distributed initialization complete")
            return fn(*args, **kwargs)

        # Then wrap with backend initializer (runs FIRST, before list_devices)
        match effective:
            case "jax":
                from skyward.integrations.jax import jax

                return jax()(inner)
            case "torch":
                from skyward.integrations.torch import torch

                return torch()(inner)
            case "tensorflow":
                from skyward.integrations.tensorflow import tensorflow

                return tensorflow()(inner)

        return inner

    return decorator
