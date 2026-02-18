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

import warnings
from collections.abc import Iterator
from concurrent.futures import Future
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal

import cloudpickle
from joblib import parallel_backend
from joblib.parallel import ParallelBackendBase, register_parallel_backend

from skyward.accelerators import Accelerator
from skyward.api.pool import ComputePool
from skyward.api.spec import DEFAULT_IMAGE, Image, InflightStrategy
from skyward.observability.logger import logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from joblib import Parallel

    from skyward.api.provider import ProviderConfig as Provider


_PROVIDER_MODULES = {"urllib3", "botocore", "boto3", "aioboto3"}


def _strip_provider_warning_filters() -> None:
    """Remove warning filters injected by cloud provider SDKs.

    sklearn.utils.parallel.Parallel captures warnings.filters and pickles
    them into every task via cloudpickle. Provider SDKs (e.g. aioboto3 →
    urllib3) register warning filters whose category classes live in modules
    not installed on workers, causing ModuleNotFoundError on deserialization.
    """
    warnings.filters[:] = [  # type: ignore[reportIndexIssue]
        f for f in warnings.filters
        if not any(
            isinstance(x, type) and x.__module__.split(".")[0] in _PROVIDER_MODULES
            for x in f
        )
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


@contextmanager
def sklearn_backend(pool: ComputePool) -> Iterator[None]:
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
    max_inflight: int | InflightStrategy | None = None,
    image: Image | None = None,
    accelerator: str | Accelerator | None = None,
    vcpus: int | None = None,
    memory_gb: int | None = None,
    allocation: Literal["spot", "on-demand", "spot-if-available"] = "spot-if-available",
    provision_timeout: int = 3600,
    joblib_version: str | None = None,
) -> Iterator[ComputePool]:
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
        provision_timeout: Provisioning timeout in seconds.
        joblib_version: Specific joblib version (e.g., "1.3.0"). None for latest.

    Yields:
        The active ComputePool.

    Example:
        with JoblibPool(provider=sky.AWS(), nodes=4) as pool:
            results = Parallel(n_jobs=-1)(delayed(fn)(x) for x in data)
    """
    logger.info(f"Creating JoblibPool with {nodes} nodes")
    base = image or DEFAULT_IMAGE
    pkg = f"joblib=={joblib_version}" if joblib_version else "joblib"
    merged = _merge_pip(base, pkg)

    pool = ComputePool(
        provider=provider,
        nodes=nodes,
        concurrency=concurrency,
        max_inflight=max_inflight,
        image=merged,
        accelerator=accelerator,
        vcpus=vcpus,
        memory_gb=memory_gb,
        allocation=allocation,
        provision_timeout=provision_timeout,

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
    max_inflight: int | InflightStrategy | None = None,
    image: Image | None = None,
    accelerator: str | Accelerator | None = None,
    vcpus: int | None = None,
    memory_gb: int | None = None,
    allocation: Literal["spot", "on-demand", "spot-if-available"] = "spot-if-available",
    provision_timeout: int = 3600,
    sklearn_version: str | None = None,
) -> Iterator[ComputePool]:
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
        provision_timeout: Provisioning timeout in seconds.
        sklearn_version: Specific sklearn version (e.g., "1.4.0"). None for latest.

    Yields:
        The active ComputePool.

    Example:
        with ScikitLearnPool(provider=sky.AWS(), nodes=4) as pool:
            grid = GridSearchCV(model, params, n_jobs=-1)
            grid.fit(X, y)  # Distributed!
    """
    logger.info(f"Creating ScikitLearnPool with {nodes} nodes")
    base = image or DEFAULT_IMAGE
    pkg = f"scikit-learn=={sklearn_version}" if sklearn_version else "scikit-learn"
    merged = _merge_pip(base, pkg)

    pool = ComputePool(
        provider=provider,
        nodes=nodes,
        concurrency=concurrency,
        max_inflight=max_inflight,
        image=merged,
        accelerator=accelerator,
        vcpus=vcpus,
        memory_gb=memory_gb,
        allocation=allocation,
        provision_timeout=provision_timeout,

    )

    with pool:
        _setup_backend(pool)
        _strip_provider_warning_filters()
        with parallel_backend("skyward"):
            yield pool
