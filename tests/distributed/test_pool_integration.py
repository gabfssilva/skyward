def test_pool_creates_registry():
    from skyward.facade import SyncComputePool

    pool = SyncComputePool.__new__(SyncComputePool)
    pool._registry = None

    assert hasattr(pool, "_registry")


def test_pool_has_collection_methods():
    from skyward.facade import SyncComputePool

    assert hasattr(SyncComputePool, "dict")
    assert hasattr(SyncComputePool, "set")
    assert hasattr(SyncComputePool, "counter")
    assert hasattr(SyncComputePool, "queue")
    assert hasattr(SyncComputePool, "barrier")
    assert hasattr(SyncComputePool, "lock")
