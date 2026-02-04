import pytest
import ray

from skyward.distributed.actors import DictActor
from skyward.distributed.proxies import DictProxy


@pytest.fixture(scope="module", autouse=True)
def ray_init():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


def test_dict_setitem_getitem():
    """Dict set and get items."""
    actor = DictActor.options(name="test:dict:basic").remote()
    proxy = DictProxy(actor, consistency="strong")

    proxy["key1"] = "value1"
    assert proxy["key1"] == "value1"

    proxy["key2"] = 42
    assert proxy["key2"] == 42

    ray.kill(actor)


def test_dict_contains():
    """Dict membership test."""
    actor = DictActor.options(name="test:dict:contains").remote()
    proxy = DictProxy(actor, consistency="strong")

    proxy["exists"] = True
    assert "exists" in proxy
    assert "missing" not in proxy

    ray.kill(actor)


def test_dict_len():
    """Dict length."""
    actor = DictActor.options(name="test:dict:len").remote()
    proxy = DictProxy(actor, consistency="strong")

    assert len(proxy) == 0
    proxy["a"] = 1
    proxy["b"] = 2
    assert len(proxy) == 2

    ray.kill(actor)


def test_dict_delete():
    """Dict delete item."""
    actor = DictActor.options(name="test:dict:del").remote()
    proxy = DictProxy(actor, consistency="strong")

    proxy["key"] = "value"
    del proxy["key"]
    assert "key" not in proxy

    ray.kill(actor)


def test_dict_get_default():
    """Dict get with default."""
    actor = DictActor.options(name="test:dict:get").remote()
    proxy = DictProxy(actor, consistency="strong")

    assert proxy.get("missing") is None
    assert proxy.get("missing", "default") == "default"
    proxy["key"] = "value"
    assert proxy.get("key") == "value"

    ray.kill(actor)


def test_dict_update():
    """Dict update multiple keys."""
    actor = DictActor.options(name="test:dict:update").remote()
    proxy = DictProxy(actor, consistency="strong")

    proxy.update({"a": 1, "b": 2, "c": 3})
    assert proxy["a"] == 1
    assert proxy["b"] == 2
    assert proxy["c"] == 3

    ray.kill(actor)


def test_dict_keys_values_items():
    """Dict keys, values, items."""
    actor = DictActor.options(name="test:dict:kvi").remote()
    proxy = DictProxy(actor, consistency="strong")

    proxy.update({"x": 10, "y": 20})

    assert set(proxy.keys()) == {"x", "y"}
    assert set(proxy.values()) == {10, 20}
    assert set(proxy.items()) == {("x", 10), ("y", 20)}

    ray.kill(actor)


def test_dict_clear():
    """Dict clear all items."""
    actor = DictActor.options(name="test:dict:clear").remote()
    proxy = DictProxy(actor, consistency="strong")

    proxy.update({"a": 1, "b": 2})
    proxy.clear()
    assert len(proxy) == 0

    ray.kill(actor)


def test_dict_pop():
    """Dict pop item."""
    actor = DictActor.options(name="test:dict:pop").remote()
    proxy = DictProxy(actor, consistency="strong")

    proxy["key"] = "value"
    result = proxy.pop("key")
    assert result == "value"
    assert "key" not in proxy
    assert proxy.pop("missing", "default") == "default"

    ray.kill(actor)
