"""Context variables and the ``sky`` singleton for implicit pool dispatch.

The ``sky`` object acts as a proxy to the currently active pool, enabling
the operator-based API (``task() >> sky``, ``task() @ sky``) without an
explicit pool reference.  Context variables track the active pool and
session so that nested ``with`` blocks resolve correctly.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

_active_pool: ContextVar[Any] = ContextVar("active_pool", default=None)
_active_session: ContextVar[Any] = ContextVar("active_session", default=None)


def get_session() -> Any | None:
    """Return the active session, or ``None`` if no session is entered."""
    return _active_session.get()


def _get_active_pool() -> Any:
    """Get the active pool from context."""
    pool = _active_pool.get()
    if pool is None:
        raise RuntimeError(
            "No active pool. Use within a @pool decorated function or 'with pool(...):' block."
        )
    return pool


class _Sky:
    """Singleton that captures >> and @ operators.

    This allows the v1-style API:
        result = compute_fn(args) >> sky   # execute on one node
        results = compute_fn(args) @ sky   # broadcast to all nodes
    """

    def __rrshift__(self, pending: Any) -> Any:
        """pending >> sky - execute computation(s)."""
        from skyward.api.function import PendingFunctionGroup

        pool = _get_active_pool()
        match pending:
            case PendingFunctionGroup():
                return pool.run_parallel(pending)
            case _:
                return pool.run(pending)

    def __rmatmul__(self, pending: Any) -> list[Any]:
        """pending @ sky - broadcast to all nodes."""
        pool = _get_active_pool()
        return pool.broadcast(pending)

    def _run_async(self, pending: Any) -> Any:
        """Submit a pending function for asynchronous execution.

        Resolves the active pool from context and delegates to
        ``pool.run_async()``.  Behind the ``task() > sky`` operator.

        Parameters
        ----------
        pending
            A ``PendingFunction`` to submit.

        Returns
        -------
        Future[T]
            A future that resolves to the remote function's return value.
        """
        pool = _get_active_pool()
        return pool.run_async(pending)

    def __repr__(self) -> str:
        pool = _active_pool.get()
        if pool:
            return f"<sky: active pool with {pool._specs[0].nodes} nodes>"
        return "<sky: no active pool>"


sky = _Sky()
