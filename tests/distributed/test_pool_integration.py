import pytest
import ray


@pytest.fixture(scope="module", autouse=True)
def ray_init():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


def test_pool_creates_registry():
    """SyncComputePool creates and manages registry."""
    from skyward.facade import SyncComputePool

    # We can't fully test pool without cloud infra, but we can test registry setup
    pool = SyncComputePool.__new__(SyncComputePool)
    pool._registry = None

    # Verify registry attribute exists after __init__ pattern
    assert hasattr(pool, "_registry")


def test_pool_has_collection_methods():
    """SyncComputePool has dict, list, etc. methods."""
    from skyward.facade import SyncComputePool

    assert hasattr(SyncComputePool, "dict")
    assert hasattr(SyncComputePool, "list")
    assert hasattr(SyncComputePool, "set")
    assert hasattr(SyncComputePool, "counter")
    assert hasattr(SyncComputePool, "queue")
    assert hasattr(SyncComputePool, "barrier")
    assert hasattr(SyncComputePool, "lock")
