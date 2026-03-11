"""Skyward plugins for third-party integrations.

All imports are lazy to avoid requiring optional dependencies at import time.

Usage:
    import skyward as sky

    with sky.ComputePool(
        plugins=[sky.plugins.torch(backend="nccl")],
    ) as compute:
        ...
"""

from typing import TYPE_CHECKING, Any

from skyward.api.plugin import AccelerateConfig as AccelerateConfig
from skyward.api.plugin import DeepSpeedConfig as DeepSpeedConfig
from skyward.api.plugin import FsdpConfig as FsdpConfig
from skyward.api.plugin import LaunchCommand as LaunchCommand
from skyward.api.plugin import LaunchContext as LaunchContext
from skyward.api.plugin import Plugin as Plugin
from skyward.api.plugin import around_app as around_app
from skyward.api.plugin import around_client as around_client
from skyward.api.plugin import around_process as around_process

if TYPE_CHECKING:
    from .accelerate import accelerate
    from .cuml import cuml
    from .jax import jax
    from .joblib import joblib
    from .keras import keras
    from .mig import mig
    from .mps import mps
    from .sklearn import sklearn
    from .torch import torch

__all__ = [
    "Plugin",
    "LaunchCommand",
    "LaunchContext",
    "accelerate",
    "torch",
    "jax",
    "keras",
    "cuml",
    "joblib",
    "mig",
    "mps",
    "sklearn",
    "around_client",
    "around_app",
    "around_process",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "accelerate": ("skyward.plugins.accelerate", "accelerate"),
    "torch": ("skyward.plugins.torch", "torch"),
    "jax": ("skyward.plugins.jax", "jax"),
    "keras": ("skyward.plugins.keras", "keras"),
    "cuml": ("skyward.plugins.cuml", "cuml"),
    "joblib": ("skyward.plugins.joblib", "joblib"),
    "mig": ("skyward.plugins.mig", "mig"),
    "mps": ("skyward.plugins.mps", "mps"),
    "sklearn": ("skyward.plugins.sklearn", "sklearn"),
}


if not TYPE_CHECKING:
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
