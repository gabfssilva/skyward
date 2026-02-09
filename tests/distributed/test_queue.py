def test_queue_put_get(registry):
    proxy = registry.queue("test_queue_basic")

    proxy.put("first")
    proxy.put("second")
    assert proxy.get() == "first"
    assert proxy.get() == "second"


def test_queue_len(registry):
    proxy = registry.queue("test_queue_len")

    assert len(proxy) == 0
    proxy.put(1)
    proxy.put(2)
    assert len(proxy) == 2
    proxy.get()
    assert len(proxy) == 1


def test_queue_empty(registry):
    proxy = registry.queue("test_queue_empty")

    assert proxy.empty()
    proxy.put("item")
    assert not proxy.empty()


def test_queue_get_timeout(registry):
    proxy = registry.queue("test_queue_timeout")

    result = proxy.get(timeout=0.1)
    assert result is None
