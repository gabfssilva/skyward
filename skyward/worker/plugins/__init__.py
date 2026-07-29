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

from skyward.shared.errors import UnsupportedPluginError
from skyward.shared.schemas import Image, PluginRef
from skyward.worker.api import Info
from skyward.worker.plugins.accelerate import Accelerate
from skyward.worker.plugins.cuml import Cuml
from skyward.worker.plugins.huggingface import HuggingFace
from skyward.worker.plugins.jax import Jax
from skyward.worker.plugins.joblib import Joblib
from skyward.worker.plugins.keras import Keras
from skyward.worker.plugins.mig import Mig
from skyward.worker.plugins.mps import Mps
from skyward.worker.plugins.plugin import Plugin
from skyward.worker.plugins.sklearn import Sklearn
from skyward.worker.plugins.torch import Torch

__all__ = ["PLUGINS", "Accelerate", "Cuml", "HuggingFace", "Jax", "Joblib", "Keras", "Mig", "Mps", "Plugin", "Sklearn", "Torch", "chain", "image", "resolve"]

PLUGINS: dict[str, type[Plugin]] = {
    Torch.kind: Torch,
    HuggingFace.kind: HuggingFace,
    Joblib.kind: Joblib,
    Jax.kind: Jax,
    Keras.kind: Keras,
    Cuml.kind: Cuml,
    Sklearn.kind: Sklearn,
    Accelerate.kind: Accelerate,
    Mig.kind: Mig,
    Mps.kind: Mps,
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
