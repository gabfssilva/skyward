import pytest
import ray

from skyward.distributed.actors import QueueActor
from skyward.distributed.proxies import QueueProxy


@pytest.fixture(scope="module", autouse=True)
def ray_init():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


def test_queue_put_get():
    """Queue put and get."""
    actor = QueueActor.options(name="test:queue:basic").remote()
    proxy = QueueProxy(actor)

    proxy.put("first")
    proxy.put("second")
    assert proxy.get() == "first"
    assert proxy.get() == "second"

    ray.kill(actor)


def test_queue_len():
    """Queue length."""
    actor = QueueActor.options(name="test:queue:len").remote()
    proxy = QueueProxy(actor)

    assert len(proxy) == 0
    proxy.put(1)
    proxy.put(2)
    assert len(proxy) == 2
    proxy.get()
    assert len(proxy) == 1

    ray.kill(actor)


def test_queue_empty():
    """Queue empty check."""
    actor = QueueActor.options(name="test:queue:empty").remote()
    proxy = QueueProxy(actor)

    assert proxy.empty()
    proxy.put("item")
    assert not proxy.empty()

    ray.kill(actor)


def test_queue_get_timeout():
    """Queue get with timeout returns None when empty."""
    actor = QueueActor.options(name="test:queue:timeout").remote()
    proxy = QueueProxy(actor)

    result = proxy.get(timeout=0.1)
    assert result is None

    ray.kill(actor)
