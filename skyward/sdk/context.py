"""The active pool, and the word that stands in for it.

``sky`` is what a call is dispatched to when the pool is left unnamed:
``train(data) >> sky`` runs on whatever pool the enclosing ``with Compute(...)``
block opened. It is not the pool — it is a stand-in resolved at dispatch time
from a context variable the block sets, which is what lets blocks nest and each
``sky`` mean the nearest one still open.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skyward.sdk.function import Pool

_active: ContextVar[Pool | None] = ContextVar("skyward_active_pool", default=None)


def enter(pool: Pool) -> Token[Pool | None]:
    """Make *pool* the active one; the returned token undoes exactly this."""
    return _active.set(pool)


def reset(token: Token[Pool | None]) -> None:
    """Restore whatever pool was active before the matching :func:`enter`."""
    _active.reset(token)


def current() -> Pool:
    """The active pool, or an error that says which one is missing.

    Raises
    ------
    RuntimeError
        When ``sky`` is dispatched to with no ``with Compute(...)`` block open.
    """
    pool = _active.get()
    if pool is None:
        raise RuntimeError(
            "`sky` has no pool to run on: dispatch to `sky` only inside a `with Compute(...)` block, "
            "or name the pool explicitly with `task() >> pool`.",
        )
    return pool


class _Sky:
    """The implicit target. One value of it exists — :data:`sky` — and an
    operator handed it resolves the active pool instead of a named one."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "sky"


sky = _Sky()
