def test_counter_increment(registry):
    proxy = registry.counter("test_counter_inc")

    assert proxy.value == 0
    proxy.increment()
    assert proxy.value == 1
    proxy.increment(5)
    assert proxy.value == 6


def test_counter_decrement(registry):
    proxy = registry.counter("test_counter_dec")

    proxy.increment(10)
    proxy.decrement()
    assert proxy.value == 9
    proxy.decrement(4)
    assert proxy.value == 5


def test_counter_reset(registry):
    proxy = registry.counter("test_counter_reset")

    proxy.increment(100)
    proxy.reset()
    assert proxy.value == 0
    proxy.reset(50)
    assert proxy.value == 50


def test_counter_int_conversion(registry):
    proxy = registry.counter("test_counter_int")

    proxy.increment(42)
    assert int(proxy) == 42
