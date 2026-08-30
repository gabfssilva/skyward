"""Joblib, with the nodes standing in for its worker processes.

A joblib ``Parallel(n_jobs=-1)(...)`` normally forks a pool of local processes.
This points it at a Skyward backend instead: each batch joblib would hand a worker
is dispatched to the compute as a task, so the fan-out is over machines, not cores.

The redirection is a client-side affair — the backend runs where ``Parallel`` is
called and reaches back into the live pool — which is why it lives in ``client``
and not in a hook that travels. The one thing the nodes need is joblib itself, so
they can unpickle the batch, and that is what ``image`` puts there.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, ClassVar

from msgspec.structs import replace

from skyward.shared.schemas import Image
from skyward.worker.plugins.plugin import Plugin

if TYPE_CHECKING:
    from skyward.core.compute import Compute


def _execute(batch: Callable[[], object]) -> object:
    """The batch, run on a node. A plain function so it pickles by reference."""
    return batch()


class Joblib(Plugin, frozen=True):
    """Install joblib on the nodes, and route its parallel backend through the pool.

    Attributes
    ----------
    version : str | None
        Pin, if the code needs one. Otherwise whatever the index has.
    """

    kind: ClassVar[str] = "joblib"

    version: str | None = None

    def image(self, image: Image) -> Image:
        package = f"joblib=={self.version}" if self.version else "joblib"
        return replace(image, pip=(*image.pip, package))

    @contextmanager
    def client(self, compute: Compute) -> Iterator[None]:
        from joblib import parallel_backend, register_parallel_backend

        backend = _backend_class()
        register_parallel_backend("skyward", lambda: backend(compute))
        with parallel_backend("skyward"):
            yield


_BACKEND: type | None = None


def _backend_class() -> type:
    """The joblib backend, defined lazily so the module imports without joblib."""
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND

    from concurrent.futures import Future

    from joblib._parallel_backends import SequentialBackend
    from joblib.parallel import ParallelBackendBase

    from skyward.core.function import Pending

    class SkywardBackend(ParallelBackendBase):
        supports_retrieve_callback = True
        supports_timeout = True  # pyright: ignore[reportIncompatibleMethodOverride, reportAssignmentType]
        uses_threads = False

        def __init__(self, compute: Compute, nesting_level: int = 0, **kwargs: object) -> None:
            super().__init__(nesting_level=nesting_level, **kwargs)
            self._compute = compute

        def effective_n_jobs(self, n_jobs: int) -> int:  # pyright: ignore[reportIncompatibleMethodOverride]
            if n_jobs == 0:
                return 0
            spec = self._compute._spec
            nodes = spec.nodes.max or spec.nodes.initial
            return nodes * (spec.worker.concurrency or 1)

        def configure(self, n_jobs: int = 1, parallel: object = None, **kwargs: object) -> int:  # pyright: ignore[reportIncompatibleMethodOverride]
            return self.effective_n_jobs(n_jobs)

        def submit(
            self,
            func: Callable[[], object],
            callback: Callable[[Future[object]], None] | None = None,
        ) -> Future[object]:
            future = self._compute.start(Pending(_execute, (func,), {}))
            if callback is not None:
                future.add_done_callback(callback)
            return future

        def retrieve_result_callback(self, out: Future[object]) -> object:  # pyright: ignore[reportIncompatibleMethodOverride]
            return out.result()

        def get_nested_backend(self) -> tuple[ParallelBackendBase, int | None]:  # pyright: ignore[reportIncompatibleMethodOverride]
            return SequentialBackend(nesting_level=(self.nesting_level or 0) + 1), None

        def terminate(self) -> None:
            pass

        def abort_everything(self, ensure_ready: bool = True) -> None:
            pass

    _BACKEND = SkywardBackend
    return _BACKEND
