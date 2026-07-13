from unittest.mock import MagicMock

import pytest

from skyward.core.pool import ComputePool

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


def test_snapshot_inactive_raises() -> None:
    pool = ComputePool.__new__(ComputePool)
    pool._active = False
    pool._pool = None
    with pytest.raises(RuntimeError, match="Pool is not active"):
        pool.snapshot()


def test_snapshot_delegates_to_pool() -> None:
    pool = ComputePool.__new__(ComputePool)
    pool._active = True
    inner = MagicMock()
    sentinel = object()
    inner.snapshot.return_value = sentinel
    pool._pool = inner

    assert pool.snapshot() is sentinel
    inner.snapshot.assert_called_once_with()
