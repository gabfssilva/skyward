"""Skyward plugins for third-party integrations.

All imports are lazy to avoid requiring optional dependencies at import time.

Usage:
    import skyward as sky

    with sky.ComputePool(
        plugins=[sky.plugins.torch(backend="nccl")],
    ) as pool:
        ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .plugin import Plugin

if TYPE_CHECKING:
    from .cuml import cuml
    from .huggingface import huggingface
    from .jax import jax
    from .joblib import joblib
    from .keras import keras
    from .mig import mig
    from .mps import mps
    from .sklearn import sklearn
    from .torch import torch

__all__ = [
    "Plugin",
    "torch",
    "jax",
    "keras",
    "cuml",
    "huggingface",
    "joblib",
    "mig",
    "mps",
    "sklearn",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "torch": ("skyward.plugins.torch", "torch"),
    "jax": ("skyward.plugins.jax", "jax"),
    "keras": ("skyward.plugins.keras", "keras"),
    "cuml": ("skyward.plugins.cuml", "cuml"),
    "huggingface": ("skyward.plugins.huggingface", "huggingface"),
    "joblib": ("skyward.plugins.joblib", "joblib"),
    "mig": ("skyward.plugins.mig", "mig"),
    "mps": ("skyward.plugins.mps", "mps"),
    "sklearn": ("skyward.plugins.sklearn", "sklearn"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        import importlib
        import sys

        module_path, attr = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        value = getattr(module, attr)
        setattr(sys.modules[__name__], name, value)
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
