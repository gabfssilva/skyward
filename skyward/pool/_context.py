"""Context management for implicit pool execution.

This module provides the ContextVar that enables the `>> sky` syntax
by tracking the currently active pool in the execution context.

Example:
    @sky.pool(provider=AWS(), nodes=4)
    def main():
        result = train(data) >> sky  # Uses pool from context
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skyward.pool.compute import ComputePool

_current_pool: ContextVar[ComputePool | None] = ContextVar("current_pool", default=None)


def get_current_pool() -> ComputePool:
    """Get the current pool from context.

    Returns:
        The active ComputePool.

    Raises:
        RuntimeError: If no pool is active in the current context.
    """
    if (pool := _current_pool.get()) is None:
        raise RuntimeError(
            "No active pool in context. "
            "Use @sky.pool decorator or 'with ComputePool(...) as pool:'"
        )
    return pool


def set_current_pool(pool: ComputePool) -> object:
    """Set the current pool in context.

    Args:
        pool: The pool to set as current.

    Returns:
        Token that can be used to reset the context.
    """
    return _current_pool.set(pool)


def reset_current_pool(token: object) -> None:
    """Reset the current pool context.

    Args:
        token: Token from set_current_pool.
    """
    _current_pool.reset(token)  # type: ignore[arg-type]
