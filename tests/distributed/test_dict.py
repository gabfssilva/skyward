def test_dict_setitem_getitem(registry):
    proxy = registry.dict("test_dict_basic")

    proxy["key1"] = "value1"
    assert proxy["key1"] == "value1"

    proxy["key2"] = 42
    assert proxy["key2"] == 42


def test_dict_contains(registry):
    proxy = registry.dict("test_dict_contains")

    proxy["exists"] = True
    assert "exists" in proxy
    assert "missing" not in proxy


def test_dict_delete(registry):
    proxy = registry.dict("test_dict_del")

    proxy["key"] = "value"
    del proxy["key"]
    assert "key" not in proxy


def test_dict_get_default(registry):
    proxy = registry.dict("test_dict_get")

    assert proxy.get("missing") is None
    assert proxy.get("missing", "default") == "default"
    proxy["key"] = "value"
    assert proxy.get("key") == "value"


def test_dict_update(registry):
    proxy = registry.dict("test_dict_update")

    proxy.update({"a": 1, "b": 2, "c": 3})
    assert proxy["a"] == 1
    assert proxy["b"] == 2
    assert proxy["c"] == 3


def test_dict_pop(registry):
    proxy = registry.dict("test_dict_pop")

    proxy["key"] = "value"
    result = proxy.pop("key")
    assert result == "value"
    assert "key" not in proxy
    assert proxy.pop("missing", "default") == "default"
