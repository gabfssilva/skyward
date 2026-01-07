"""Keras 3 distributed training integration."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Literal

__all__ = ["keras"]

Backend = Literal["jax", "torch", "tensorflow"] | None


def keras[**P, R](
    backend: Backend = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Configure Keras 3 distributed training.

    Args:
        backend: Backend to use (jax, torch, tensorflow). Defaults to jax.

    Example:
        >>> import skyward as sky

        >>> @sky.compute
        ... @sky.integrations.keras(backend="jax")
        ... def train():
        ...     import keras
        ...     model = keras.Sequential([...])
        ...     model.fit(...)
    """

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        effective = backend or "jax"

        # Apply backend decorator (lazy import)
        match effective:
            case "jax":
                from skyward.integrations.jax import jax

                fn = jax()(fn)
            case "torch":
                from skyward.integrations.torch import torch

                fn = torch()(fn)
            case "tensorflow":
                from skyward.integrations.tensorflow import tensorflow

                fn = tensorflow()(fn)

        @functools.wraps(fn)
        def inner(*args: P.args, **kwargs: P.kwargs) -> R:
            from skyward.cluster import instance_info

            pool = instance_info()

            if pool and pool.total_nodes > 1:
                import keras

                keras.utils.set_random_seed(42)
                devices = keras.distribution.list_devices()

                if devices:
                    keras.distribution.set_distribution(
                        keras.distribution.DataParallel(
                            devices=devices,
                            auto_shard_dataset=False,
                        )
                    )

            return fn(*args, **kwargs)

        return inner

    return decorator
