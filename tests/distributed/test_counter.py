import pytest
import ray

from skyward.distributed.actors import CounterActor
from skyward.distributed.proxies import CounterProxy


@pytest.fixture(scope="module", autouse=True)
def ray_init():
    """Initialize Ray for tests."""
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


def test_counter_increment():
    """Counter increments correctly."""
    actor = CounterActor.options(name="test:counter:inc").remote()
    proxy = CounterProxy(actor, consistency="strong")

    assert proxy.value == 0
    proxy.increment()
    assert proxy.value == 1
    proxy.increment(5)
    assert proxy.value == 6

    ray.kill(actor)


def test_counter_decrement():
    """Counter decrements correctly."""
    actor = CounterActor.options(name="test:counter:dec").remote()
    proxy = CounterProxy(actor, consistency="strong")

    proxy.increment(10)
    proxy.decrement()
    assert proxy.value == 9
    proxy.decrement(4)
    assert proxy.value == 5

    ray.kill(actor)


def test_counter_reset():
    """Counter resets correctly."""
    actor = CounterActor.options(name="test:counter:reset").remote()
    proxy = CounterProxy(actor, consistency="strong")

    proxy.increment(100)
    proxy.reset()
    assert proxy.value == 0
    proxy.reset(50)
    assert proxy.value == 50

    ray.kill(actor)


def test_counter_int_conversion():
    """Counter converts to int."""
    actor = CounterActor.options(name="test:counter:int").remote()
    proxy = CounterProxy(actor, consistency="strong")

    proxy.increment(42)
    assert int(proxy) == 42

    ray.kill(actor)
