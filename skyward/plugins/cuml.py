"""RAPIDS cuML, with scikit-learn quietly running on the GPU."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import ClassVar

from msgspec.structs import replace

from skyward.plugins.plugin import Plugin
from skyward.protocol.schemas import Image, PipIndex
from skyward.runtime.api import Info

_lock = threading.Lock()
_installed = False
"""Whether ``cuml.accel`` has patched this process. Process-global, because the
patch is a monkeypatch of scikit-learn's estimators and doing it twice is waste,
not a no-op."""


class Cuml(Plugin, frozen=True):
    """Install cuML from NVIDIA's index, and put its sklearn accelerator on before
    the first task runs.

    The accelerator is installed in the process that runs the task, on that
    process's first task, not at worker start. ``cuml.accel.install()`` rewrites
    scikit-learn's estimators in the interpreter that imports them, and under a
    subprocess executor that is the child, not the worker; installing it in the
    worker would leave the child running plain CPU sklearn. Doing it on the first
    task, once and under a lock, is what lets the same plugin serve either
    executor.

    Attributes
    ----------
    cuda : str
        The CUDA suffix of the RAPIDS wheel to install, e.g. ``cu12``. Names the
        package (``cuml-cu12``) as much as the build.
    """

    kind: ClassVar[str] = "cuml"

    cuda: str = "cu12"

    def image(self, image: Image) -> Image:
        package = f"cuml-{self.cuda}"
        return replace(
            image,
            pip=(*image.pip, package),
            pip_indexes=(*image.pip_indexes, PipIndex(url="https://pypi.nvidia.com", packages=(package,))),
        )

    def run[T](self, call: Callable[[], T], info: Info) -> T:
        _install()
        return call()


def _install() -> None:
    """Turn on cuML's sklearn acceleration, once for this process.

    The double check keeps the import and the patch off the fast path once it is
    on: every task after the first sees the flag and calls straight through.
    """
    global _installed
    if _installed:
        return

    with _lock:
        if _installed:
            return

        import cuml.accel

        cuml.accel.install()
        _installed = True
