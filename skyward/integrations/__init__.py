"""Skyward integrations with third-party libraries.

All imports are lazy to avoid requiring optional dependencies at import time.

Available integrations:

Distributed training:
- keras: Keras 3 distributed training decorator
- torch: PyTorch distributed training decorator
- jax: JAX distributed training decorator
- tensorflow: TensorFlow distributed training decorator
- transformers: Hugging Face Transformers distributed training decorator

Joblib/sklearn:
- JoblibPool: Distributed joblib execution
- ScikitLearnPool: Distributed sklearn training
- sklearn_backend: Low-level backend for existing ComputePool

Usage:
    from skyward import AWS, compute
    from skyward.integrations import keras, JoblibPool

    @keras(backend="jax")
    @compute
    def train():
        ...

    with JoblibPool(provider=AWS(), nodes=4):
        results = Parallel(n_jobs=-1)(delayed(fn)(x) for x in data)
"""

from typing import Any

__all__ = [
    # Distributed training
    "keras",
    "torch",
    "jax",
    "tensorflow",
    "transformers",
    # Joblib/sklearn
    "JoblibPool",
    "ScikitLearnPool",
    "sklearn_backend",
    "joblib_backend",
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
    "sklearn_backend": ("skyward.integrations.joblib", "sklearn_backend"),
    "joblib_backend": ("skyward.integrations.joblib", "joblib_backend"),
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
