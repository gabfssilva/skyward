import pytest
import ray
import threading
import time

from skyward.distributed.actors import BarrierActor
from skyward.distributed.proxies import BarrierProxy


@pytest.fixture(scope="module", autouse=True)
def ray_init():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


def test_barrier_wait():
    """Barrier releases when n parties arrive."""
    actor = BarrierActor.options(name="test:barrier:basic").remote(n=2)
    proxy1 = BarrierProxy(actor)
    proxy2 = BarrierProxy(actor)

    results = []
    errors = []

    def wait_and_record(proxy, name):
        try:
            proxy.wait()
            results.append(name)
        except Exception as e:
            errors.append(str(e))

    t1 = threading.Thread(target=wait_and_record, args=(proxy1, "t1"))
    t2 = threading.Thread(target=wait_and_record, args=(proxy2, "t2"))

    # Start both threads close together
    t1.start()
    t2.start()

    # Wait for completion with longer timeout
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert len(errors) == 0, f"Errors: {errors}"
    assert len(results) == 2, f"Results: {results}, threads still alive: t1={t1.is_alive()}, t2={t2.is_alive()}"

    ray.kill(actor)


def test_barrier_reset():
    """Barrier can be reset and reused."""
    actor = BarrierActor.options(name="test:barrier:reset").remote(n=1)
    proxy = BarrierProxy(actor)

    proxy.wait()  # First use
    proxy.reset()
    proxy.wait()  # Second use after reset

    ray.kill(actor)
