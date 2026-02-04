import pytest
import ray

from skyward.distributed.actors import SetActor
from skyward.distributed.proxies import SetProxy


@pytest.fixture(scope="module", autouse=True)
def ray_init():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


def test_set_add_contains():
    """Set add and contains."""
    actor = SetActor.options(name="test:set:basic").remote()
    proxy = SetProxy(actor, consistency="strong")

    proxy.add("item1")
    assert "item1" in proxy
    assert "item2" not in proxy

    ray.kill(actor)


def test_set_len():
    """Set length."""
    actor = SetActor.options(name="test:set:len").remote()
    proxy = SetProxy(actor, consistency="strong")

    assert len(proxy) == 0
    proxy.add("a")
    proxy.add("b")
    proxy.add("a")  # duplicate
    assert len(proxy) == 2

    ray.kill(actor)


def test_set_discard():
    """Set discard."""
    actor = SetActor.options(name="test:set:discard").remote()
    proxy = SetProxy(actor, consistency="strong")

    proxy.add("item")
    proxy.discard("item")
    assert "item" not in proxy
    proxy.discard("missing")  # should not raise

    ray.kill(actor)


def test_set_clear():
    """Set clear."""
    actor = SetActor.options(name="test:set:clear").remote()
    proxy = SetProxy(actor, consistency="strong")

    proxy.add("a")
    proxy.add("b")
    proxy.clear()
    assert len(proxy) == 0

    ray.kill(actor)
