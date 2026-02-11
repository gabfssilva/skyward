from skyward.distributed.proxies import CounterProxy, DictProxy


def test_registry_get_or_create_dict(registry):
    d1 = registry.dict("test_dict")
    assert isinstance(d1, DictProxy)


def test_registry_get_or_create_counter(registry):
    c1 = registry.counter("test_counter")
    assert isinstance(c1, CounterProxy)


def test_registry_consistency_override(registry):
    d1 = registry.dict("cons_dict", consistency="strong")
    assert d1._consistency == "strong"

    d2 = registry.dict("cons_dict2", consistency="eventual")
    assert d2._consistency == "eventual"
