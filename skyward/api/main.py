"""``@sky.main`` entry point decorator."""
from __future__ import annotations

from collections.abc import Callable
from functools import wraps


def main[**P, T](fn: Callable[P, T] | None = None) -> Callable[P, T]:
    """Mark a function as a Skyward entry point.

    Can be used as ``@sky.main`` or ``@sky.main()``.
    The decorated function runs locally and orchestrates
    pool operations via ``>> sky`` / ``@ sky``.
    """
    if fn is None:
        return main  # type: ignore[return-value]

    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        return fn(*args, **kwargs)

    wrapper.__sky_main__ = True  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]
