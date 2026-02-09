"""Skyward joblib backend for sklearn integration.

Enables distributed sklearn/joblib workloads with minimal code changes:

    import skyward as sky
    from skyward.integrations.joblib import JoblibPool
    from joblib import Parallel, delayed

    with JoblibPool(provider=sky.AWS(), nodes=4) as pool:
        results = Parallel(n_jobs=-1)(delayed(fn)(x) for x in data)

Or with sklearn:

    from skyward.integrations.joblib import ScikitLearnPool

    with ScikitLearnPool(provider=sky.AWS(), nodes=4) as pool:
        grid = GridSearchCV(model, params, n_jobs=-1)
        grid.fit(X, y)  # Distributed!

Uses joblib's pluggable backend system for seamless integration.
"""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import Future
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal

import cloudpickle
from joblib import parallel_backend
from joblib.parallel import ParallelBackendBase, register_parallel_backend
from loguru import logger

from skyward.accelerators import Accelerator
from skyward.facade import SyncComputePool
from skyward.image import DEFAULT_IMAGE, Image

if TYPE_CHECKING:
    from collections.abc import Callable

    from joblib import Parallel

    from skyward.providers import AWS, RunPod, VastAI, Verda

    type Provider = AWS | VastAI | Verda | RunPod


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

    def __init__(self, pool: SyncComputePool, nesting_level: int = 0, **kwargs: Any):
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
        return self.pool.nodes * self.pool.concurrency

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


def _setup_backend(pool: SyncComputePool) -> None:
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


@contextmanager
def sklearn_backend(pool: SyncComputePool) -> Iterator[None]:
    """Context manager to use Skyward as the joblib backend.

    Any sklearn code with `n_jobs=-1` will automatically distribute
    work to the Skyward pool.
    """
    logger.info("Activating sklearn/joblib backend with Skyward pool")
    _setup_backend(pool)
    with parallel_backend("skyward"):
        yield
    logger.debug("sklearn/joblib backend deactivated")


joblib_backend = sklearn_backend


def _merge_pip(base: Image, *packages: str) -> Image:
    return Image(
        python=base.python,
        pip=[*base.pip, *packages],
        pip_extra_index_url=base.pip_extra_index_url,
        apt=base.apt,
        env=base.env,
        skyward_source=base.skyward_source,
    )


@contextmanager
def JoblibPool(
    provider: Provider,
    *,
    nodes: int = 1,
    concurrency: int = 1,
    image: Image | None = None,
    accelerator: str | Accelerator | None = None,
    vcpus: int | None = None,
    memory_gb: int | None = None,
    allocation: Literal["spot", "on-demand", "spot-if-available"] = "spot-if-available",
    timeout: int = 3600,
    panel: bool = True,
    joblib_version: str | None = None,
) -> Iterator[SyncComputePool]:
    """Compute pool with joblib backend for distributed parallel execution.

    Automatically adds joblib to pip dependencies and sets up the backend.

    Args:
        provider: Cloud provider (AWS, Verda, VastAI, RunPod).
        nodes: Number of nodes to provision.
        image: Base image. joblib added automatically.
        accelerator: GPU/accelerator specification.
        vcpus: CPU cores per worker.
        memory_gb: Memory per worker in GB.
        allocation: Instance allocation strategy.
        timeout: Provisioning timeout in seconds.
        panel: Enable Rich terminal dashboard.
        joblib_version: Specific joblib version (e.g., "1.3.0"). None for latest.

    Yields:
        The active SyncComputePool.

    Example:
        with JoblibPool(provider=sky.AWS(), nodes=4) as pool:
            results = Parallel(n_jobs=-1)(delayed(fn)(x) for x in data)
    """
    logger.info(f"Creating JoblibPool with {nodes} nodes")
    base = image or DEFAULT_IMAGE
    pkg = f"joblib=={joblib_version}" if joblib_version else "joblib"
    merged = _merge_pip(base, pkg)

    pool = SyncComputePool(
        provider=provider,
        nodes=nodes,
        concurrency=concurrency,
        image=merged,
        accelerator=accelerator,
        vcpus=vcpus,
        memory_gb=memory_gb,
        allocation=allocation,
        timeout=timeout,
        panel=panel,
    )

    with pool:
        _setup_backend(pool)
        with parallel_backend("skyward"):
            yield pool


@contextmanager
def ScikitLearnPool(
    provider: Provider,
    *,
    nodes: int = 1,
    concurrency: int = 1,
    image: Image | None = None,
    accelerator: str | Accelerator | None = None,
    vcpus: int | None = None,
    memory_gb: int | None = None,
    allocation: Literal["spot", "on-demand", "spot-if-available"] = "spot-if-available",
    timeout: int = 3600,
    panel: bool = True,
    sklearn_version: str | None = None,
) -> Iterator[SyncComputePool]:
    """Compute pool with scikit-learn for distributed ML training.

    Automatically adds scikit-learn to pip dependencies and sets up the backend.

    Args:
        provider: Cloud provider (AWS, Verda, VastAI, RunPod).
        nodes: Number of nodes to provision.
        image: Base image. scikit-learn added automatically.
        accelerator: GPU/accelerator specification.
        vcpus: CPU cores per worker.
        memory_gb: Memory per worker in GB.
        allocation: Instance allocation strategy.
        timeout: Provisioning timeout in seconds.
        panel: Enable Rich terminal dashboard.
        sklearn_version: Specific sklearn version (e.g., "1.4.0"). None for latest.

    Yields:
        The active SyncComputePool.

    Example:
        with ScikitLearnPool(provider=sky.AWS(), nodes=4) as pool:
            grid = GridSearchCV(model, params, n_jobs=-1)
            grid.fit(X, y)  # Distributed!
    """
    logger.info(f"Creating ScikitLearnPool with {nodes} nodes")
    base = image or DEFAULT_IMAGE
    pkg = f"scikit-learn=={sklearn_version}" if sklearn_version else "scikit-learn"
    merged = _merge_pip(base, pkg)

    pool = SyncComputePool(
        provider=provider,
        nodes=nodes,
        concurrency=concurrency,
        image=merged,
        accelerator=accelerator,
        vcpus=vcpus,
        memory_gb=memory_gb,
        allocation=allocation,
        timeout=timeout,
        panel=panel,
    )

    with pool:
        _setup_backend(pool)
        with parallel_backend("skyward"):
            yield pool
