"""Joblib plugin — distributed parallel execution backend."""

from __future__ import annotations

import sys
import warnings
from collections.abc import Iterator
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import cloudpickle
from joblib import parallel_backend
from joblib.parallel import ParallelBackendBase, register_parallel_backend

from skyward.observability.logger import logger
from skyward.plugins.plugin import Plugin

if TYPE_CHECKING:
    from collections.abc import Callable

    from joblib import Parallel

    from skyward.api.model import Cluster
    from skyward.api.pool import ComputePool
    from skyward.api.spec import Image


_SAFE_ROOTS = sys.stdlib_module_names | {"builtins"}


def _strip_local_warning_filters() -> None:
    """Keep only stdlib/builtin warning filters for safe cross-environment serialization.

    sklearn.utils.parallel.Parallel captures ``warnings.filters`` and pickles
    them into every task via cloudpickle.  Filters whose category classes come
    from third-party modules (pytest, cloud SDKs, …) reference modules that may
    not exist on workers, causing ``ModuleNotFoundError`` on deserialization.

    Only stdlib/builtin categories (``DeprecationWarning``, ``FutureWarning``,
    …) are guaranteed to exist everywhere.  Third-party packages installed on
    workers will re-inject their own filters at import time.
    """
    warnings.filters[:] = [  # type: ignore[reportIndexIssue]
        f for f in warnings.filters
        if f[2].__module__.split(".")[0] in _SAFE_ROOTS
    ]


def _make_remote_executor() -> Callable[[bytes], Any]:
    from skyward import compute

    @compute
    def execute_batch(payload: bytes) -> Any:
        import cloudpickle

        batch_func = cloudpickle.loads(payload)
        return batch_func()

    return execute_batch


class SkywardBackend(ParallelBackendBase):
    supports_retrieve_callback = True  # type: ignore[assignment]
    supports_inner_max_num_threads = True  # type: ignore[assignment]
    uses_threads = False  # type: ignore[assignment]
    supports_timeout = True  # type: ignore[assignment]

    def __init__(self, pool: ComputePool, nesting_level: int = 0, **kwargs: Any) -> None:
        super().__init__(nesting_level=nesting_level, **kwargs)
        self.pool = pool
        self.parallel: Parallel | None = None
        self._n_jobs = 1
        self._remote_execute = _make_remote_executor()

    def configure(  # type: ignore[override]
        self,
        n_jobs: int = 1,
        parallel: Parallel | None = None,
        **kwargs: Any,
    ) -> int:
        self.parallel = parallel
        self._n_jobs = n_jobs
        effective = self.effective_n_jobs(n_jobs)
        logger.debug(f"Configuring SkywardBackend with n_jobs={n_jobs}, effective={effective}")
        return effective

    def terminate(self) -> None:
        pass

    def effective_n_jobs(self, n_jobs: int) -> int:  # type: ignore[override]
        if n_jobs == 0:
            return 0
        return self.pool._specs[0].nodes * self.pool.concurrency

    def submit(
        self,
        func: Callable[[], Any],
        callback: Callable[[Future[Any]], None] | None = None,
    ) -> Future[Any]:
        payload = cloudpickle.dumps(func)
        pending = self._remote_execute(payload)
        future: Future[Any] = pending > self.pool
        if callback is not None:
            future.add_done_callback(callback)
        return future

    def retrieve_result_callback(self, future: Future[Any]) -> Any:  # type: ignore[override]
        return future.result()

    def get_nested_backend(self) -> tuple[ParallelBackendBase, int]:  # type: ignore[override]
        from joblib._parallel_backends import SequentialBackend

        return SequentialBackend(nesting_level=self.nesting_level + 1), None  # type: ignore[return-value]

    def abort_everything(self, ensure_ready: bool = True) -> None:
        pass


_backend_registered = False


def _setup_backend(pool: ComputePool) -> None:
    global _backend_registered

    def backend_factory() -> SkywardBackend:
        return SkywardBackend(pool)

    if not _backend_registered:
        logger.debug("Registering Skyward joblib backend")
        register_parallel_backend("skyward", backend_factory)
        _backend_registered = True
    else:
        logger.debug("Re-registering Skyward joblib backend")
        register_parallel_backend("skyward", backend_factory, make_default=False)


def joblib(version: str | None = None) -> Plugin:
    """Joblib plugin with Skyward parallel backend.

    Parameters
    ----------
    version
        Specific joblib version (e.g. "1.3.0"). None for latest.
    """

    def transform(image: Image, cluster: Cluster[Any]) -> Image:
        pkg = f"joblib=={version}" if version else "joblib"
        return replace(image, pip=(*image.pip, pkg))

    @contextmanager
    def around_client(pool: ComputePool, cluster: Cluster[Any]) -> Iterator[None]:
        _setup_backend(pool)
        _strip_local_warning_filters()
        with parallel_backend("skyward"):
            yield

    return (
        Plugin.create("joblib")
        .with_image_transform(transform)
        .with_around_client(around_client)
    )
