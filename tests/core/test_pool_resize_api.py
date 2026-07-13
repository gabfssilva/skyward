from unittest.mock import MagicMock

import pytest

from skyward.api.spec import Nodes
from skyward.core.pool import ComputePool


def _active_pool_with_fake_pool() -> tuple[ComputePool, MagicMock]:
    """Build a ComputePool shell with _active=True and a mock pool object.

    Avoids spinning a real pool — we only need to verify that resize()
    pattern-matches and schedules ``Pool.resize`` with the right payload.
    """
    pool = ComputePool.__new__(ComputePool)
    pool._active = True
    inner = MagicMock()
    pool._pool = inner
    loop = MagicMock()
    loop.call_soon_threadsafe = lambda fn, *args: fn(*args)
    pool._loop = loop
    return pool, inner


class TestResizeAPI:
    def test_int_form(self) -> None:
        pool, inner = _active_pool_with_fake_pool()
        pool.resize(3)
        inner.resize.assert_called_once_with(Nodes(desired=3))

    def test_tuple_form(self) -> None:
        pool, inner = _active_pool_with_fake_pool()
        pool.resize(2, 5)
        inner.resize.assert_called_once_with(Nodes(desired=2, max=5))

    def test_nodes_form(self) -> None:
        pool, inner = _active_pool_with_fake_pool()
        pool.resize(Nodes(desired=4, min=2, max=8))
        inner.resize.assert_called_once_with(Nodes(desired=4, min=2, max=8))

    def test_inactive_pool_raises(self) -> None:
        pool = ComputePool.__new__(ComputePool)
        pool._active = False
        pool._pool = None
        with pytest.raises(RuntimeError, match="Pool is not active"):
            pool.resize(2)

    def test_invalid_form_raises(self) -> None:
        pool, _ = _active_pool_with_fake_pool()
        with pytest.raises(TypeError, match="resize"):
            pool.resize()
        with pytest.raises(TypeError, match="resize"):
            pool.resize(1, 2, 3)
