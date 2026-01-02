"""Skyward integrations with third-party libraries.

Available integrations:
- joblib: Distributed sklearn via joblib backend

Usage:
    from skyward.integrations import sklearn_backend

Note: Integrations are lazy-loaded. You need the corresponding
library installed (e.g., scikit-learn for sklearn_backend).
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skyward.integrations.joblib import sklearn_backend as sklearn_backend

__all__ = ["sklearn_backend"]


def __getattr__(name: str):
    """Lazy import integrations to avoid requiring optional dependencies."""
    if name == "sklearn_backend":
        try:
            from skyward.integrations.joblib import sklearn_backend

            return sklearn_backend
        except ImportError as e:
            raise ImportError(
                "sklearn_backend requires joblib. "
                "Install it with: pip install scikit-learn"
            ) from e

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
