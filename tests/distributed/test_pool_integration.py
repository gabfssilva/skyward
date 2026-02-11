def test_pool_creates_registry():
    from skyward.api.pool import ComputePool

    pool = ComputePool.__new__(ComputePool)
    pool._registry = None

    assert hasattr(pool, "_registry")


def test_pool_has_collection_methods():
    from skyward.api.pool import ComputePool

    assert hasattr(ComputePool, "dict")
    assert hasattr(ComputePool, "set")
    assert hasattr(ComputePool, "counter")
    assert hasattr(ComputePool, "queue")
    assert hasattr(ComputePool, "barrier")
    assert hasattr(ComputePool, "lock")
