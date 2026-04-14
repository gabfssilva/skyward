"""Tests for the named plugin registry."""

from __future__ import annotations

import pytest

import skyward.server.wire as wire
from skyward.api.plugin import Plugin
from skyward.plugins.torch import torch
from skyward.server.wire import (
    UnknownPluginTag,
    UnserializablePlugin,
    decode_plugin,
    encode_plugin,
    from_dict,
    is_registered_plugin,
    register_plugin,
    registered_plugin_names,
    to_dict,
)


def test_builtin_plugins_registered() -> None:
    names = registered_plugin_names()
    for expected in ("torch", "jax", "keras", "cuml", "joblib", "sklearn", "mig", "mps"):
        assert expected in names, f"missing {expected} in {names}"


def test_torch_plugin_encode() -> None:
    plugin = torch(backend="gloo", cuda="cu124")
    encoded = encode_plugin(plugin)
    assert encoded == {
        "type": "torch",
        "args": {"backend": "gloo", "cuda": "cu124"},
    }


def test_torch_plugin_roundtrip_via_codec() -> None:
    plugin = torch(backend="gloo", cuda="cu124")
    encoded = to_dict(plugin)
    assert encoded["type"] == "torch"
    decoded = from_dict(encoded, Plugin)
    assert isinstance(decoded, Plugin)
    assert decoded.name == plugin.name
    # Decoder produces a fresh plugin; the decorator (when present) must
    # behave identically because the same factory was invoked with the
    # same kwargs.  Compare a concrete observable attribute.
    assert (decoded.decorate is None) == (plugin.decorate is None)


def test_decode_plugin_direct() -> None:
    obj = {"type": "jax", "args": {"cuda": "cu124"}}
    decoded = decode_plugin(obj)
    assert isinstance(decoded, Plugin)
    assert decoded.name == "jax"
    assert is_registered_plugin(decoded)


def test_decode_unknown_tag_raises() -> None:
    with pytest.raises(UnknownPluginTag):
        decode_plugin({"type": "bogus", "args": {}})


def test_inline_plugin_raises_unserializable() -> None:
    inline = Plugin.create("custom").with_decorator(lambda fn: fn)
    with pytest.raises(UnserializablePlugin) as exc:
        encode_plugin(inline)
    assert "custom" in str(exc.value)


def test_to_dict_on_inline_plugin_raises() -> None:
    inline = Plugin.create("custom2").with_decorator(lambda fn: fn)
    with pytest.raises(UnserializablePlugin):
        to_dict(inline)


def test_register_plugin_installs_into_skyward_plugins_namespace() -> None:
    # After wire import, calling skyward.plugins.torch(...) must produce
    # a registered plugin (not an inline unregistered one).
    from skyward import plugins as sky_plugins

    plugin = sky_plugins.torch(backend="gloo", cuda="cu124")
    assert is_registered_plugin(plugin)


def test_register_plugin_accepts_schema_kwarg() -> None:
    # Schema is forward-compatibility only.
    def fake_factory(**kwargs: object) -> Plugin:
        return Plugin(name="fake")

    wrapped = register_plugin("fake-test-schema", fake_factory, schema=None, install=False)
    plugin = wrapped()
    assert is_registered_plugin(plugin)
    assert encode_plugin(plugin) == {"type": "fake-test-schema", "args": {}}


def test_wire_module_exports() -> None:
    for name in (
        "register",
        "register_encoder",
        "register_plugin",
        "to_dict",
        "from_dict",
        "UnserializablePlugin",
        "UnknownPluginTag",
        "UnknownWireType",
    ):
        assert hasattr(wire, name)
