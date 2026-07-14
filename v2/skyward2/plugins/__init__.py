"""The plugins, and the two ends that put them to work.

Registered by name. The name is what travels — the spec carries ``PluginRef`` and
never an object — so a node rebuilds the plugin from its parameters rather than
being sent one, and a kind nobody has heard of is refused at the door instead of
crashing a worker an hour later.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial, reduce

from msgspec import ValidationError, convert

from skyward2.application.errors import UnsupportedPluginError
from skyward2.plugins.huggingface import HuggingFace
from skyward2.plugins.plugin import Plugin
from skyward2.plugins.torch import Torch
from skyward2.protocol.schemas import Image, PluginRef
from skyward2.runtime.api import Info

__all__ = ["PLUGINS", "HuggingFace", "Plugin", "Torch", "chain", "image", "resolve"]

PLUGINS: dict[str, type[Plugin]] = {
    Torch.kind: Torch,
    HuggingFace.kind: HuggingFace,
}


def resolve(refs: Sequence[PluginRef]) -> tuple[Plugin, ...]:
    """The plugins a spec asked for, or a refusal naming the one that does not exist.

    The parameters are validated here, against the plugin's own fields. A misspelt
    backend is an error the user gets back from the call that created the compute,
    rather than a traceback out of a worker on a machine they are already paying for.
    """

    def one(ref: PluginRef) -> Plugin:
        plugin = PLUGINS.get(ref.kind)
        if plugin is None:
            raise UnsupportedPluginError(f"no plugin named {ref.kind}", kind=ref.kind)

        try:
            return convert(ref.params, type=plugin)
        except ValidationError as invalid:
            raise UnsupportedPluginError(f"{ref.kind}: {invalid}", kind=ref.kind) from invalid

    return tuple(one(ref) for ref in refs)


def image(base: Image, plugins: Sequence[Plugin]) -> Image:
    """The image the plugins want, each handed what the ones before it asked for."""
    return reduce(lambda current, plugin: plugin.image(current), plugins, base)


def chain[T](
    plugins: Sequence[Plugin],
    call: Callable[[], T],
    info: Info,
) -> Callable[[], T]:
    """The call, wrapped by every plugin, the first one outermost."""
    wrapped: Callable[[], T] = call
    for plugin in reversed(plugins):
        wrapped = partial(plugin.run, wrapped, info)
    return wrapped
