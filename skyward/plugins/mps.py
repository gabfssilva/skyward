"""CUDA MPS, started once per worker so a GPU serves several tasks at a time."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import ClassVar

from skyward.plugins.plugin import Plugin
from skyward.runtime.api import Info

PIPE_DIRECTORY = "/tmp/nvidia-mps"
LOG_DIRECTORY = "/tmp/nvidia-mps-log"


class Mps(Plugin, frozen=True):
    """Bring up the MPS control daemon before the worker takes its first task.

    Multi-Process Service lets several CUDA processes share one GPU context instead
    of each taking the whole device, which is what a worker running tasks in parallel
    needs the GPU to allow. The daemon is started in ``setup``, in the worker process
    and before any child is spawned, so every task inherits the pipe directory it
    rendezvous on. The image cannot carry it: MPS ships with the CUDA driver, so
    there is nothing to install, and ``env`` reaches only the bootstrap shell that has
    since exited — the daemon and its variables have to be put up where the tasks run.

    Starting the daemon is best-effort. On a machine without the control binary the
    call fails and is swallowed, because a worker that cannot share its GPU should
    still run the task on the whole one, not refuse to start.

    Attributes
    ----------
    active_thread_percentage : int | None
        Ceiling on the share of GPU compute one client may use, 1-100. Left to MPS's
        default when unset.
    pinned_memory_limit : str | None
        Per-device pinned memory limit, e.g. ``"0=2G"`` for 2 GB on device 0.
    """

    kind: ClassVar[str] = "mps"
    collective: ClassVar[bool] = False

    active_thread_percentage: int | None = None
    pinned_memory_limit: str | None = None

    @contextmanager
    def setup(self, info: Info) -> Iterator[None]:
        os.makedirs(PIPE_DIRECTORY, exist_ok=True)
        os.makedirs(LOG_DIRECTORY, exist_ok=True)
        with suppress(OSError):
            subprocess.run(["nvidia-cuda-mps-control", "-d"], check=False)

        os.environ["CUDA_MPS_PIPE_DIRECTORY"] = PIPE_DIRECTORY
        os.environ["CUDA_MPS_LOG_DIRECTORY"] = LOG_DIRECTORY
        if self.active_thread_percentage is not None:
            os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(self.active_thread_percentage)
        if self.pinned_memory_limit is not None:
            os.environ["CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"] = self.pinned_memory_limit
        yield
