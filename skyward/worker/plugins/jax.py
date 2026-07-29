"""JAX, already a process cluster by the time the user's function runs."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import ClassVar

from msgspec.structs import replace

from skyward.shared.schemas import Image, PipIndex
from skyward.worker.api import Info
from skyward.worker.plugins.plugin import Plugin

PORT = 1234

_lock = threading.Lock()
_joined = False
"""Whether this process has already joined the cluster. Process-global, because a
process initializes jax's distributed runtime once and doing it twice is an error,
not a no-op."""


class Jax(Plugin, frozen=True):
    """Install jax, and join the process cluster before the first task runs it.

    The rendezvous is rank zero, because jax insists on being told where the
    coordinator is — the compute has no head, and this is the convention that
    satisfies a library that believes it does.

    It is joined in the process that runs the task, and on that process's first
    task, not at worker start. ``jax.distributed.initialize`` is a collective —
    every node blocks in it until the last one arrives — and the node that arrives
    there must be the one that will run the collective code afterwards. Under a
    subprocess executor that is the child, not the worker; joining in the worker
    would leave the child outside a cluster it never entered. Doing it on the first
    task, once and under a lock, is what lets the same plugin serve either executor.

    Attributes
    ----------
    cuda : str
        The CUDA build to install jax from, as a ``jax[...]`` extra. Pinned rather
        than left to the default, because the wheel is matched to the driver the GPU
        images ship.
    """

    kind: ClassVar[str] = "jax"
    collective: ClassVar[bool] = True

    cuda: str = "cu124"

    def image(self, image: Image) -> Image:
        return replace(
            image,
            pip=(*image.pip, f"jax[{self.cuda}]"),
            pip_indexes=(
                *image.pip_indexes,
                PipIndex(
                    url="https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
                    packages=("jax", "jaxlib"),
                ),
            ),
        )

    def run[T](self, call: Callable[[], T], info: Info) -> T:
        _join(info)
        return call()


def _join(info: Info) -> None:
    """Join jax's distributed runtime, once for this process.

    The double check keeps the import and the collective off the fast path once the
    cluster is up: every task after the first sees the flag and calls straight through.
    """
    global _joined
    if _joined:
        return

    with _lock:
        if _joined:
            return

        import jax

        jax.distributed.initialize(
            coordinator_address=f"{info.head}:{PORT}",
            num_processes=info.nodes,
            process_id=info.rank,
        )
        _joined = True
