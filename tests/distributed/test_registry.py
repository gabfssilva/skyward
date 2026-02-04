import pytest
import ray

from skyward.distributed.registry import DistributedRegistry
from skyward.distributed.proxies import DictProxy, CounterProxy


@pytest.fixture(scope="module", autouse=True)
def ray_init():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.fixture
def registry():
    reg = DistributedRegistry()
    yield reg
    reg.cleanup()


def test_registry_get_or_create_dict(registry):
    """Registry creates dict on first access."""
    d1 = registry.dict("test_dict")
    assert isinstance(d1, DictProxy)

    d2 = registry.dict("test_dict")
    assert d1._actor == d2._actor  # Same actor


def test_registry_get_or_create_counter(registry):
    """Registry creates counter on first access."""
    c1 = registry.counter("test_counter")
    assert isinstance(c1, CounterProxy)

    c2 = registry.counter("test_counter")
    assert c1._actor == c2._actor  # Same actor


def test_registry_different_names(registry):
    """Different names create different actors."""
    d1 = registry.dict("dict_a")
    d2 = registry.dict("dict_b")
    assert d1._actor != d2._actor


def test_registry_consistency_override(registry):
    """Registry respects consistency override."""
    d1 = registry.dict("cons_dict", consistency="strong")
    assert d1._consistency == "strong"

    d2 = registry.dict("cons_dict2", consistency="eventual")
    assert d2._consistency == "eventual"


def test_registry_cleanup(registry):
    """Registry cleanup destroys all actors."""
    registry.dict("cleanup_dict")
    registry.counter("cleanup_counter")

    registry.cleanup()

    # After cleanup, actors should be gone
    # Next access creates new actors
    d = registry.dict("cleanup_dict")
    d["key"] = "value"
    assert d["key"] == "value"
