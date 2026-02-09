def test_set_add_contains(registry):
    proxy = registry.set("test_set_basic")

    proxy.add("item1")
    assert "item1" in proxy
    assert "item2" not in proxy


def test_set_len(registry):
    proxy = registry.set("test_set_len")

    assert len(proxy) == 0
    proxy.add("a")
    proxy.add("b")
    proxy.add("a")  # duplicate
    assert len(proxy) == 2


def test_set_discard(registry):
    proxy = registry.set("test_set_discard")

    proxy.add("item")
    proxy.discard("item")
    assert "item" not in proxy
    proxy.discard("missing")  # should not raise
