import pytest
import ray
import threading
import time

from skyward.distributed.actors import LockActor
from skyward.distributed.proxies import LockProxy


@pytest.fixture(scope="module", autouse=True)
def ray_init():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


def test_lock_acquire_release():
    """Lock acquire and release."""
    actor = LockActor.options(name="test:lock:basic").remote()
    proxy = LockProxy(actor)

    assert proxy.acquire()
    proxy.release()

    ray.kill(actor)


def test_lock_context_manager():
    """Lock as context manager."""
    actor = LockActor.options(name="test:lock:ctx").remote()
    proxy = LockProxy(actor)

    with proxy:
        pass  # Lock held here

    # Lock released, can acquire again
    assert proxy.acquire()
    proxy.release()

    ray.kill(actor)


def test_lock_mutual_exclusion():
    """Lock provides mutual exclusion."""
    actor = LockActor.options(name="test:lock:mutex").remote()
    # Each thread needs its own proxy with unique holder_id
    proxy1 = LockProxy(actor)
    proxy2 = LockProxy(actor)

    results = []

    def critical_section(proxy, name):
        with proxy:
            results.append(f"{name}_enter")
            time.sleep(0.1)
            results.append(f"{name}_exit")

    t1 = threading.Thread(target=critical_section, args=(proxy1, "t1"))
    t2 = threading.Thread(target=critical_section, args=(proxy2, "t2"))

    t1.start()
    time.sleep(0.01)  # Let t1 enter first
    t2.start()

    t1.join()
    t2.join()

    # Should be sequential: t1_enter, t1_exit, t2_enter, t2_exit
    # or t2_enter, t2_exit, t1_enter, t1_exit
    assert results[0].endswith("_enter")
    assert results[1].endswith("_exit")
    assert results[0][:2] == results[1][:2]  # Same thread

    ray.kill(actor)
