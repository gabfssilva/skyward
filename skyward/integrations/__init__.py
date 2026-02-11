"""Skyward v2 integrations with third-party libraries.

All imports are lazy to avoid requiring optional dependencies at import time.

Available integrations:

Distributed training:
- keras: Keras 3 distributed training decorator
- torch: PyTorch distributed training decorator
- jax: JAX distributed training decorator
- tensorflow: TensorFlow distributed training decorator
- transformers: Hugging Face Transformers distributed training decorator

Parallel execution:
- JoblibPool: Distributed joblib execution
- ScikitLearnPool: Distributed scikit-learn training

Usage:
    from skyward import compute
    from skyward.integrations import keras

    @keras(backend="jax")
    @compute
    def train():
        ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .jax import jax
    from .joblib import JoblibPool, ScikitLearnPool
    from .keras import keras
    from .tensorflow import tensorflow
    from .torch import torch
    from .transformers import transformers

__all__ = [
    # Distributed training
    "keras",
    "torch",
    "jax",
    "tensorflow",
    "transformers",
    # Parallel execution
    "JoblibPool",
    "ScikitLearnPool",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # (module, attribute)
    "keras": ("skyward.integrations.keras", "keras"),
    "torch": ("skyward.integrations.torch", "torch"),
    "jax": ("skyward.integrations.jax", "jax"),
    "tensorflow": ("skyward.integrations.tensorflow", "tensorflow"),
    "transformers": ("skyward.integrations.transformers", "transformers"),
    "JoblibPool": ("skyward.integrations.joblib", "JoblibPool"),
    "ScikitLearnPool": ("skyward.integrations.joblib", "ScikitLearnPool"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        import sys

        module = importlib.import_module(module_path)
        value = getattr(module, attr)

        # Cache in module globals to prevent subsequent __getattr__ calls
        # and override any submodule reference Python may have added
        setattr(sys.modules[__name__], name, value)

        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
