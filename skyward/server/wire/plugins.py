"""Named plugin registry for wire serialization.

Plugins hold callables (``ImageTransform``, ``TaskDecorator``,
``ClientLifecycle``, etc.) that do not JSON-serialize.  We encode them by
**factory name + kwargs** and reconstruct identical instances on the
receiving side by re-invoking the registered factory.

Identity is tracked in a ``WeakValueDictionary`` keyed by ``id(plugin)``:
registered factories are wrapped so every produced :class:`Plugin` is
recorded with its name and encoded constructor kwargs.  Inline plugins
built via :class:`Plugin.create` therefore raise
:class:`UnserializablePlugin` at encode time.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from skyward.api.plugin import Plugin

__all__ = [
    "PluginFactory",
    "UnknownPluginTag",
    "UnserializablePlugin",
    "decode_plugin",
    "encode_plugin",
    "is_registered_plugin",
    "register_plugin",
    "registered_plugin_names",
]


type PluginFactory = Callable[..., Plugin]


_FACTORIES: dict[str, PluginFactory] = {}
_ORIGINAL: dict[str, PluginFactory] = {}
_NAME_BY_ID: dict[int, str] = {}
_ARGS_BY_ID: dict[int, dict[str, Any]] = {}
# Strong references; Plugin is a slotted frozen dataclass without
# ``__weakref__`` so ``WeakValueDictionary`` is not usable.  The set of
# registered instances stays small (one per registered plugin call) for the
# lifetime of a process, which is acceptable.
_KEEPALIVE: list[Plugin] = []


class UnserializablePlugin(Exception):  # noqa: N818 — public name mandated by spec
    """Raised when a :class:`Plugin` was not produced by a registered factory."""


class UnknownPluginTag(Exception):  # noqa: N818 — public name mandated by spec
    """Raised when decoding a plugin with an unregistered ``type`` tag."""


def register_plugin(
    name: str,
    factory: PluginFactory,
    *,
    schema: type | None = None,
    install: bool = True,
) -> PluginFactory:
    """Register a plugin factory by name.

    Parameters
    ----------
    name
        Wire tag used in ``{"type": name, "args": {...}}``.
    factory
        Callable returning a :class:`Plugin`.  Wrapped so every produced
        plugin records its identity in the registry.
    schema
        Optional pydantic model class describing accepted kwargs.
        Currently unused; kept for forward compatibility.
    install
        When ``True`` (the default), replace ``factory`` inside its owning
        module so any caller that resolves ``sky.plugins.<name>`` invokes
        the wrapped version and the produced plugin is registered.

    Returns
    -------
    PluginFactory
        The wrapped factory.  Safe to call in place of the original.
    """
    del schema
    _ORIGINAL[name] = factory

    @functools.wraps(factory)
    def wrapped(**kwargs: Any) -> Plugin:
        plugin = factory(**kwargs)
        _NAME_BY_ID[id(plugin)] = name
        _ARGS_BY_ID[id(plugin)] = dict(kwargs)
        _KEEPALIVE.append(plugin)
        return plugin

    _FACTORIES[name] = wrapped
    if install:
        _install_factory(factory, wrapped)
    return wrapped


def _install_factory(original: PluginFactory, wrapped: PluginFactory) -> None:
    """Replace *original* with *wrapped* inside its owning module(s).

    Installs into both the plugin's own submodule (e.g.
    ``skyward.plugins.torch.torch``) and into ``skyward.plugins`` (shadowing
    any lazy ``__getattr__`` result).
    """
    import sys

    module_name = getattr(original, "__module__", None)
    attr_name = getattr(original, "__name__", None)
    if module_name is None or attr_name is None:
        return
    owner = sys.modules.get(module_name)
    if owner is not None and getattr(owner, attr_name, None) is original:
        setattr(owner, attr_name, wrapped)
    parent = sys.modules.get("skyward.plugins")
    if parent is not None:
        setattr(parent, attr_name, wrapped)


def is_registered_plugin(plugin: Plugin) -> bool:
    """Return whether *plugin* was produced by a registered factory."""
    return id(plugin) in _NAME_BY_ID


def registered_plugin_names() -> tuple[str, ...]:
    """Return the sorted tuple of registered plugin tags."""
    return tuple(sorted(_FACTORIES))


def encode_plugin(plugin: Plugin) -> dict[str, Any]:
    """Encode *plugin* as ``{"type": name, "args": {...}}``.

    Parameters
    ----------
    plugin
        Plugin instance produced by a registered factory.

    Returns
    -------
    dict[str, Any]
        Wire-format dict.

    Raises
    ------
    UnserializablePlugin
        If *plugin* was not produced by a registered factory.
    """
    if not is_registered_plugin(plugin):
        raise UnserializablePlugin(
            f"plugin {plugin.name!r} was not produced by a registered factory; "
            "inline plugins (Plugin.create(...).with_*()) cannot be serialized"
        )
    name = _NAME_BY_ID[id(plugin)]
    args = _ARGS_BY_ID[id(plugin)]
    return {"type": name, "args": dict(args)}


def decode_plugin(obj: dict[str, Any]) -> Plugin:
    """Decode a wire-format plugin dict.

    Parameters
    ----------
    obj
        Dict of the shape ``{"type": name, "args": {...}}``.

    Returns
    -------
    Plugin
        A fresh plugin instance produced by invoking the registered factory.

    Raises
    ------
    UnknownPluginTag
        If ``obj["type"]`` is not a registered name.
    """
    match obj:
        case {"type": str(name), "args": dict(args)}:
            factory = _FACTORIES.get(name)
            if factory is None:
                raise UnknownPluginTag(f"no plugin registered for tag {name!r}")
            return factory(**args)
        case _:
            raise ValueError(f"malformed plugin wire object: {obj!r}")
