"""Keras 3 distributed training integration."""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from skyward.integrations.jax import _init_jax, _jax_env_vars
from skyward.integrations.tensorflow import _tensorflow_env_vars
from skyward.integrations.torch import _init_pytorch, _pytorch_env_vars

if TYPE_CHECKING:
    from skyward.cluster import InstanceInfo

Backend = Literal["jax", "torch", "tensorflow"] | None

__all__ = ["keras"]

_ENV_VAR_BUILDERS: dict[str, Callable[[InstanceInfo], dict[str, str]]] = {
    "torch": _pytorch_env_vars,
    "jax": _jax_env_vars,
    "tensorflow": _tensorflow_env_vars,
}

_BACKEND_INITIALIZERS: dict[str, Callable[[InstanceInfo], None]] = {
    "torch": _init_pytorch,
    "jax": _init_jax,
}


def _init_keras(pool: InstanceInfo) -> None:
    """Initialize Keras 3 DataParallel distribution."""
    if pool.total_nodes <= 1:
        return

    import keras

    keras.utils.set_random_seed(42)

    devices = keras.distribution.list_devices()
    if not devices:
        return

    data_parallel = keras.distribution.DataParallel(devices=devices, auto_shard_dataset=False)
    keras.distribution.set_distribution(data_parallel)


def keras[**P, R](
    backend: Backend = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Configure Keras 3 distributed training.

    Args:
        backend: Backend to use (jax, torch, tensorflow). Auto-detected if None.

    Example:
        from skyward.integrations import keras

        @keras(backend="jax")
        @compute
        def train():
            import keras
            model = keras.Sequential([...])
            model.fit(...)
    """
    from skyward.pending import ComputeFunction

    def wrapper(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def inner(*args: P.args, **kwargs: P.kwargs) -> R:
            from skyward.cluster import instance_info

            pool = instance_info()

            if pool is not None:
                effective = backend if backend else "jax"

                builder = _ENV_VAR_BUILDERS.get(effective)
                if builder:
                    env_vars = builder(pool)
                    for key, value in env_vars.items():
                        os.environ[key] = value

                initializer = _BACKEND_INITIALIZERS.get(effective)
                if initializer:
                    initializer(pool)

                _init_keras(pool)

            return fn(*args, **kwargs)

        return inner

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        if isinstance(fn, ComputeFunction):
            return ComputeFunction(fn=wrapper(fn.fn), name=fn.name)
        return wrapper(fn)

    return decorator
