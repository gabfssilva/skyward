def test_dict_function(registry):
    import skyward as sky
    from skyward.distributed import _set_active_registry

    _set_active_registry(registry)

    try:
        d = sky.dict("api_test_dict")
        d["key"] = "value"
        assert d["key"] == "value"
    finally:
        _set_active_registry(None)


def test_counter_function(registry):
    import skyward as sky
    from skyward.distributed import _set_active_registry

    _set_active_registry(registry)

    try:
        c = sky.counter("api_test_counter")
        c.increment(5)
        assert c.value == 5
    finally:
        _set_active_registry(None)


def test_all_functions_exist():
    import skyward as sky

    assert hasattr(sky, "dict")
    assert hasattr(sky, "set")
    assert hasattr(sky, "counter")
    assert hasattr(sky, "queue")
    assert hasattr(sky, "barrier")
    assert hasattr(sky, "lock")
