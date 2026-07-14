"""PyTorch, already distributed by the time the user's function starts."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import ClassVar, Literal

from msgspec.structs import replace

from skyward.plugins.plugin import Plugin
from skyward.protocol.schemas import Image
from skyward.runtime.api import Info

PORT = 29500


class Torch(Plugin, frozen=True):
    """Install torch, and form the process group before the first task.

    The rendezvous is rank zero, because torch insists on being told where it is —
    the compute has no head, and this is the convention that satisfies a library
    that believes it does.

    The group is formed at worker start rather than around each task, and formed
    once: ``init_process_group`` is a collective, so every node blocks in it until
    the last one arrives, and doing that per task would pay the barrier again for
    every call.

    Attributes
    ----------
    backend : Literal["nccl", "gloo"]
        ``nccl`` on GPUs, ``gloo`` on CPUs.
    version : str | None
        Pin, if the code needs one. Otherwise whatever the index has.
    """

    kind: ClassVar[str] = "torch"
    collective: ClassVar[bool] = True

    backend: Literal["nccl", "gloo"] = "nccl"
    version: str | None = None

    def image(self, image: Image) -> Image:
        package = f"torch=={self.version}" if self.version else "torch"
        return replace(image, packages=(*image.packages, package))

    @contextmanager
    def setup(self, info: Info) -> Iterator[None]:
        import torch.distributed as dist

        os.environ.setdefault("MASTER_ADDR", info.head)
        os.environ.setdefault("MASTER_PORT", str(PORT))
        os.environ.setdefault("RANK", str(info.rank))
        os.environ.setdefault("WORLD_SIZE", str(info.nodes))

        dist.init_process_group(backend=self.backend, rank=info.rank, world_size=info.nodes)
        try:
            yield
        finally:
            dist.destroy_process_group()
