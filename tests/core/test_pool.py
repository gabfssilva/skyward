from unittest.mock import MagicMock

import pytest

import skyward.core.pool as poolmod
from skyward.actors.messages import GetPoolSnapshot
from skyward.core.pool import ComputePool

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


def test_snapshot_inactive_raises() -> None:
    pool = ComputePool.__new__(ComputePool)
    pool._active = False
    pool._pool_ref = None
    pool._system = None
    with pytest.raises(RuntimeError, match="Pool is not active"):
        pool.snapshot()


def test_snapshot_asks_pool_actor(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = ComputePool.__new__(ComputePool)
    pool._active = True
    pool._system = MagicMock()
    pool._pool_ref = MagicMock()
    pool._loop = MagicMock()

    sentinel = object()
    monkeypatch.setattr(poolmod, "run_sync", lambda _loop, _coro, **_kw: sentinel)

    assert pool.snapshot() is sentinel

    factory = pool._system.ask.call_args.args[1]
    assert isinstance(factory(MagicMock()), GetPoolSnapshot)
