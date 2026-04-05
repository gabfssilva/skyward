from pathlib import Path

from skyward.api.pool import Pool
from skyward.daemon.pool import DaemonPool


class TestDaemonPool:
    def test_satisfies_pool_protocol(self) -> None:
        pool = DaemonPool(name="train", socket_path=Path("/tmp/fake.sock"))
        assert isinstance(pool, Pool)

    def test_not_active_before_enter(self) -> None:
        pool = DaemonPool(name="train", socket_path=Path("/tmp/fake.sock"))
        assert not pool.is_active

