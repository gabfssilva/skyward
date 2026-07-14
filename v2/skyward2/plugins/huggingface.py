"""The hub, logged in before anything asks it for a gated model."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import ClassVar

from msgspec.structs import replace

from skyward2.plugins.plugin import Plugin
from skyward2.protocol.schemas import Image
from skyward2.runtime.api import Info


class HuggingFace(Plugin, frozen=True):
    """Put ``huggingface_hub`` on the machine and the token in its environment.

    The token is set in ``setup`` rather than baked into the image because the image
    is a description of a machine and the token is a secret about a person. It is
    still carried in the spec, and the spec is still written to the database in the
    clear — which is a thing to fix, and is not fixed by putting it somewhere else
    on the same machine.

    Attributes
    ----------
    token : str | None
        Read from ``HF_TOKEN`` by the SDK when it is not given.
    """

    kind: ClassVar[str] = "huggingface"

    token: str | None = None

    def image(self, image: Image) -> Image:
        return replace(image, packages=(*image.packages, "huggingface_hub"))

    @contextmanager
    def setup(self, info: Info) -> Iterator[None]:
        if self.token:
            os.environ["HF_TOKEN"] = self.token
        yield
