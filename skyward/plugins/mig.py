"""NVIDIA MIG — one physical GPU carved into slices, one slice per subprocess."""

from __future__ import annotations

import os
import re
import subprocess
import threading
from collections.abc import Callable
from typing import ClassVar

from skyward.plugins.plugin import Plugin
from skyward.protocol.schemas import Image
from skyward.runtime.api import Info
from skyward.runtime.bootstrap import phase

_DCGM_INSTALL = """\
if ! command -v dcgmi >/dev/null 2>&1; then
    apt-get install -y -qq datacenter-gpu-manager 2>/dev/null || true
fi
if command -v nv-hostengine >/dev/null 2>&1 && ! pgrep -x nv-hostengine >/dev/null 2>&1; then
    nv-hostengine 2>/dev/null || true
fi"""

_lock = threading.Lock()
_pinned = False
"""Whether this process has already claimed its MIG slice. Process-global, because
a subprocess pins one device once and re-reading ``nvidia-smi -L`` per task is waste,
not a no-op."""


class Mig(Plugin, frozen=True):
    """Partition the GPU with MIG and give each subprocess its own slice.

    The GPU is put in MIG mode and cut into ``concurrency`` instances during
    bootstrap; at run time each worker subprocess reads the slice UUIDs and pins the
    one at its index through ``CUDA_VISIBLE_DEVICES``. That indexing is the whole
    contract, and it only holds under ``executor='process'`` with ``reuse=True``:
    there each concurrent slot is a distinct, long-lived child with a stable
    ``info.worker``, so slice *k* belongs to child *k* for the child's life. Under
    the thread executor every task shares one process and one ``info.worker`` of
    zero — they would all pin the same slice — and without ``reuse`` a child dies
    after its task, so the pinning buys nothing.

    The pin is set on the process's first task, once and under a lock, because the
    process that must see ``CUDA_VISIBLE_DEVICES`` is the one that will import the
    GPU library and run the task — the child, not the worker that spawned it.

    Attributes
    ----------
    profile : str
        The MIG profile every slice is cut to, e.g. ``"3g.40gb"`` or ``"1g.10gb"``.
    """

    kind: ClassVar[str] = "mig"
    collective: ClassVar[bool] = False

    profile: str

    def bootstrap(self, image: Image, concurrency: int) -> tuple[str, ...]:
        create = tuple(f"nvidia-smi mig -cgi {self.profile} -C" for _ in range(concurrency))
        return (
            phase("nvidia-mig", "nvidia-smi -mig 1", *create),
            phase("nvidia-dcgm", _DCGM_INSTALL),
        )

    def run[T](self, call: Callable[[], T], info: Info) -> T:
        _pin(info)
        return call()


def _pin(info: Info) -> None:
    """Claim this process's MIG slice, once.

    The double check keeps the ``nvidia-smi`` call off the fast path once the slice
    is pinned: every task after the first sees the flag and calls straight through.
    """
    global _pinned
    if _pinned:
        return

    with _lock:
        if _pinned:
            return

        listing = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True)
        uuids = re.findall(r"MIG-[0-9a-fA-F-]+", listing.stdout)
        os.environ["CUDA_VISIBLE_DEVICES"] = uuids[info.worker]
        _pinned = True
