import pytest
import ray


@pytest.fixture(scope="module", autouse=True)
def ray_init():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


def test_dict_function():
    """sky.dict() function works."""
    import skyward as sky
    from skyward.distributed import _set_active_registry
    from skyward.distributed.registry import DistributedRegistry

    reg = DistributedRegistry()
    _set_active_registry(reg)

    try:
        d = sky.dict("api_test_dict")
        d["key"] = "value"
        assert d["key"] == "value"
    finally:
        reg.cleanup()
        _set_active_registry(None)


def test_counter_function():
    """sky.counter() function works."""
    import skyward as sky
    from skyward.distributed import _set_active_registry
    from skyward.distributed.registry import DistributedRegistry

    reg = DistributedRegistry()
    _set_active_registry(reg)

    try:
        c = sky.counter("api_test_counter")
        c.increment(5)
        assert c.value == 5
    finally:
        reg.cleanup()
        _set_active_registry(None)


def test_all_functions_exist():
    """All distributed functions are exported."""
    import skyward as sky

    assert hasattr(sky, "dict")
    assert hasattr(sky, "list")
    assert hasattr(sky, "set")
    assert hasattr(sky, "counter")
    assert hasattr(sky, "queue")
    assert hasattr(sky, "barrier")
    assert hasattr(sky, "lock")
