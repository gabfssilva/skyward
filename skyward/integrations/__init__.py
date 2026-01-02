"""Skyward integrations with third-party libraries.

Available integrations:
- JoblibPool: Distributed joblib execution
- ScikitLearnPool: Distributed sklearn training
- sklearn_backend: Low-level backend for existing ComputePool

Usage:
    from skyward import AWS
    from skyward.integrations import JoblibPool

    with JoblibPool(provider=AWS(), nodes=4):
        results = Parallel(n_jobs=-1)(delayed(fn)(x) for x in data)
"""

from skyward.integrations.joblib import (
    JoblibPool,
    ScikitLearnPool,
    joblib_backend,
    sklearn_backend,
)

__all__ = ["JoblibPool", "ScikitLearnPool", "sklearn_backend", "joblib_backend"]
