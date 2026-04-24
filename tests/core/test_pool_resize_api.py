from unittest.mock import MagicMock

import pytest

from skyward.actors.pool.messages import Resize
from skyward.api.spec import Nodes
from skyward.core.pool import ComputePool


def _active_pool_with_fake_ref() -> tuple[ComputePool, MagicMock]:
    """Build a ComputePool shell with _active=True and a mock pool ref.

    Avoids spinning the full actor system — we only need to verify that
    resize() pattern-matches and tells(Resize) with the right payload.
    """
    pool = ComputePool.__new__(ComputePool)
    pool._active = True
    ref = MagicMock()
    pool._pool_ref = ref
    pool._system = MagicMock()
    return pool, ref


class TestResizeAPI:
    def test_int_form(self) -> None:
        pool, ref = _active_pool_with_fake_ref()
        pool.resize(3)
        ref.tell.assert_called_once_with(Resize(nodes=Nodes(desired=3)))

    def test_tuple_form(self) -> None:
        pool, ref = _active_pool_with_fake_ref()
        pool.resize(2, 5)
        ref.tell.assert_called_once_with(Resize(nodes=Nodes(desired=2, max=5)))

    def test_nodes_form(self) -> None:
        pool, ref = _active_pool_with_fake_ref()
        pool.resize(Nodes(desired=4, min=2, max=8))
        ref.tell.assert_called_once_with(
            Resize(nodes=Nodes(desired=4, min=2, max=8)),
        )

    def test_inactive_pool_raises(self) -> None:
        pool = ComputePool.__new__(ComputePool)
        pool._active = False
        pool._pool_ref = None
        pool._system = None
        with pytest.raises(RuntimeError, match="Pool is not active"):
            pool.resize(2)

    def test_invalid_form_raises(self) -> None:
        pool, _ = _active_pool_with_fake_ref()
        with pytest.raises(TypeError, match="resize"):
            pool.resize()
        with pytest.raises(TypeError, match="resize"):
            pool.resize(1, 2, 3)
