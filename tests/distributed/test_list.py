import pytest
import ray

from skyward.distributed.actors import ListActor
from skyward.distributed.proxies import ListProxy


@pytest.fixture(scope="module", autouse=True)
def ray_init():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


def test_list_append_getitem():
    """List append and get."""
    actor = ListActor.options(name="test:list:basic").remote()
    proxy = ListProxy(actor, consistency="strong")

    proxy.append("first")
    proxy.append("second")
    assert proxy[0] == "first"
    assert proxy[1] == "second"

    ray.kill(actor)


def test_list_len():
    """List length."""
    actor = ListActor.options(name="test:list:len").remote()
    proxy = ListProxy(actor, consistency="strong")

    assert len(proxy) == 0
    proxy.append(1)
    proxy.append(2)
    assert len(proxy) == 2

    ray.kill(actor)


def test_list_extend():
    """List extend."""
    actor = ListActor.options(name="test:list:extend").remote()
    proxy = ListProxy(actor, consistency="strong")

    proxy.extend([1, 2, 3])
    assert len(proxy) == 3
    assert proxy[0] == 1
    assert proxy[2] == 3

    ray.kill(actor)


def test_list_pop():
    """List pop."""
    actor = ListActor.options(name="test:list:pop").remote()
    proxy = ListProxy(actor, consistency="strong")

    proxy.extend([1, 2, 3])
    assert proxy.pop() == 3
    assert len(proxy) == 2
    assert proxy.pop(0) == 1
    assert len(proxy) == 1

    ray.kill(actor)


def test_list_slice():
    """List slice."""
    actor = ListActor.options(name="test:list:slice").remote()
    proxy = ListProxy(actor, consistency="strong")

    proxy.extend([0, 1, 2, 3, 4, 5])
    result = proxy.slice(1, 4)
    assert result == [1, 2, 3]

    ray.kill(actor)


def test_list_clear():
    """List clear."""
    actor = ListActor.options(name="test:list:clear").remote()
    proxy = ListProxy(actor, consistency="strong")

    proxy.extend([1, 2, 3])
    proxy.clear()
    assert len(proxy) == 0

    ray.kill(actor)
