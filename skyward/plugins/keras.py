"""Keras 3, told which backend it runs on before it is imported."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import ClassVar, Literal

from msgspec.structs import replace

from skyward.plugins.plugin import Plugin
from skyward.protocol.schemas import Image
from skyward.runtime.api import Info

type Backend = Literal["jax", "tensorflow", "torch"]


class Keras(Plugin, frozen=True):
    """Install Keras and its backend, and set the backend before the first import.

    Keras reads ``KERAS_BACKEND`` once, when it is imported, and is stuck with what
    it finds — so the backend is set in ``setup``, in the worker process and before
    any task imports keras. The image cannot do it: its ``env`` reaches the bootstrap
    shell, which exits, and never the worker the tasks run in. The backend value
    doubles as its package name, which is why one field installs both.

    Multi-node training on the ``jax`` backend is data-parallel: every node runs the
    same graph over its own shard, and the only thing they must agree on is the
    random state, which ``setup`` synchronizes. Forming the JAX process group is not
    this plugin's job — pair it with the ``jax`` plugin, which is the collective and
    the one that knows the rendezvous.

    Attributes
    ----------
    backend : Literal["jax", "tensorflow", "torch"]
        The framework Keras runs on, and the package installed to carry it.
    """

    kind: ClassVar[str] = "keras"

    backend: Backend = "jax"

    def image(self, image: Image) -> Image:
        return replace(image, pip=(*image.pip, "keras", self.backend))

    @contextmanager
    def setup(self, info: Info) -> Iterator[None]:
        os.environ["KERAS_BACKEND"] = self.backend
        if info.nodes > 1 and self.backend == "jax":
            import keras

            devices = keras.distribution.list_devices()
            if devices:
                keras.distribution.set_distribution(
                    keras.distribution.DataParallel(devices=devices, auto_shard_dataset=False),
                )

                from keras.src.backend.jax.distribution_lib import initialize_rng

                initialize_rng()
        yield
