"""PyTorch, already distributed by the time the user's function runs."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from typing import ClassVar, Literal

from msgspec.structs import replace

from skyward.plugins.plugin import Plugin
from skyward.protocol.schemas import Image, PipIndex
from skyward.runtime.api import Info

PORT = 29500

_lock = threading.Lock()
_joined = False
"""Whether this process has already formed the group. Process-global, because a
process has one default group and forming it twice is an error, not a no-op."""


class Torch(Plugin, frozen=True):
    """Install torch, and form the process group before the first task runs it.

    The rendezvous is rank zero, because torch insists on being told where it is —
    the compute has no head, and this is the convention that satisfies a library
    that believes it does.

    It is formed in the process that runs the task, and on that process's first
    task, not at worker start. ``init_process_group`` is a collective — every node
    blocks in it until the last one arrives — and the node that arrives there must
    be the one that will run the collective code afterwards. Under a subprocess
    executor that is the child, not the worker; forming it in the worker would leave
    the child holding a group it never joined. Doing it on the first task, once and
    under a lock, is what lets the same plugin serve either executor.

    Attributes
    ----------
    backend : Literal["nccl", "gloo"]
        ``nccl`` on GPUs, ``gloo`` on CPUs.
    cuda : str
        The CUDA build to install torch from, as a ``download.pytorch.org/whl``
        suffix. Pinned rather than left to PyPI's default, because the default
        tracks the newest CUDA and the newest CUDA outruns the driver the GPU
        images ship — a torch built for a CUDA the driver cannot load hangs on the
        first collective, which is exactly where it looks like a network fault and
        is not. Ignored for ``gloo``, which takes the CPU wheel.
    version : str | None
        Pin, if the code needs one. Otherwise whatever the index has.
    """

    kind: ClassVar[str] = "torch"
    collective: ClassVar[bool] = True

    backend: Literal["nccl", "gloo"] = "nccl"
    cuda: str = "cu128"
    version: str | None = None

    def image(self, image: Image) -> Image:
        package = f"torch=={self.version}" if self.version else "torch"
        url = f"https://download.pytorch.org/whl/{self.cuda}" if self.backend == "nccl" else "https://download.pytorch.org/whl/cpu"
        return replace(
            image,
            pip=(*image.pip, package),
            pip_indexes=(*image.pip_indexes, PipIndex(url=url, packages=("torch",))),
        )

    def run[T](self, call: Callable[[], T], info: Info) -> T:
        _join(self.backend, info)
        return call()


def _join(backend: str, info: Info) -> None:
    """Form the default process group, once for this process.

    The double check keeps the import and the collective off the fast path once the
    group is up: every task after the first sees the flag and calls straight through.
    """
    global _joined
    if _joined:
        return

    with _lock:
        if _joined:
            return

        import torch.distributed as dist

        os.environ["MASTER_ADDR"] = info.head
        os.environ["MASTER_PORT"] = str(PORT)
        os.environ["RANK"] = str(info.rank)
        os.environ["WORLD_SIZE"] = str(info.nodes)
        os.environ["LOCAL_RANK"] = "0"
        os.environ["LOCAL_WORLD_SIZE"] = "1"
        os.environ["NODE_RANK"] = str(info.rank)

        dist.init_process_group(backend=backend, rank=info.rank, world_size=info.nodes)
        _joined = True
