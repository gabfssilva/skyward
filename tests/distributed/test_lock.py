import threading
import time


def test_lock_acquire_release(registry):
    proxy = registry.lock("test_lock_basic")

    assert proxy.acquire()
    proxy.release()


def test_lock_context_manager(registry):
    proxy = registry.lock("test_lock_ctx")

    with proxy:
        pass

    assert proxy.acquire()
    proxy.release()


def test_lock_mutual_exclusion(registry):
    proxy1 = registry.lock("test_lock_mutex")
    proxy2 = registry.lock("test_lock_mutex")

    results = []

    def critical_section(proxy, name):
        with proxy:
            results.append(f"{name}_enter")
            time.sleep(0.1)
            results.append(f"{name}_exit")

    t1 = threading.Thread(target=critical_section, args=(proxy1, "t1"))
    t2 = threading.Thread(target=critical_section, args=(proxy2, "t2"))

    t1.start()
    time.sleep(0.01)
    t2.start()

    t1.join()
    t2.join()

    assert results[0].endswith("_enter")
    assert results[1].endswith("_exit")
    assert results[0][:2] == results[1][:2]
