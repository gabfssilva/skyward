"""Skyward joblib backend for sklearn integration.

Enables distributed sklearn/joblib workloads with minimal code changes:

    from skyward import AWS
    from skyward.integrations import JoblibPool
    from joblib import Parallel, delayed

    with JoblibPool(provider=AWS(), nodes=4, concurrency=4):
        results = Parallel(n_jobs=-1)(delayed(fn)(x) for x in data)

Or with sklearn:

    from skyward.integrations import ScikitLearnPool

    with ScikitLearnPool(provider=AWS(), nodes=4):
        grid = GridSearchCV(model, params, n_jobs=-1)
        grid.fit(X, y)  # Distributed!

Uses joblib's pluggable backend system for seamless integration.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal

import cloudpickle
from joblib import parallel_backend
from joblib.parallel import ParallelBackendBase, register_parallel_backend
from loguru import logger

from skyward.accelerator import Accelerator
from skyward.callback import Callback
from skyward.image import DEFAULT_IMAGE, Image
from skyward.pool import ComputePool
from skyward.spec import AllocationLike
from skyward.types import Memory
from skyward.volume import Volume

if TYPE_CHECKING:
    from joblib import Parallel

    from skyward.types import Provider


def _make_remote_executor() -> Callable[[bytes], Any]:
    """Create the remote execution function."""
    from skyward import compute

    @compute
    def execute_batch(payload: bytes) -> Any:
        """Execute a serialized BatchedCalls on a remote instance."""
        import cloudpickle

        batch_func = cloudpickle.loads(payload)
        return batch_func()

    return execute_batch


class SkywardBackend(ParallelBackendBase):
    """Joblib backend that dispatches tasks to Skyward.

    Uses a ThreadPoolExecutor internally to provide async task submission.
    Each thread calls the blocking Skyward API, allowing parallel execution.
    """

    supports_retrieve_callback = True
    supports_inner_max_num_threads = True
    uses_threads = False
    supports_timeout = True

    def __init__(self, pool: ComputePool, nesting_level: int = 0, **kwargs: Any):
        super().__init__(nesting_level=nesting_level, **kwargs)
        self.pool = pool
        self.parallel: Parallel | None = None
        self._n_jobs = 1
        self._executor: ThreadPoolExecutor | None = None
        self._remote_execute = _make_remote_executor()

    def configure(
        self,
        n_jobs: int = 1,
        parallel: Parallel | None = None,
        prefer: str | None = None,
        require: str | None = None,
        **kwargs: Any,
    ) -> int:
        """Configure the backend for parallel execution."""
        self.parallel = parallel
        self._n_jobs = n_jobs
        effective = self.effective_n_jobs(n_jobs)
        logger.debug(f"Configuring SkywardBackend with n_jobs={n_jobs}, effective={effective}")
        self._executor = ThreadPoolExecutor(max_workers=effective)
        return effective

    def terminate(self) -> None:
        """Shutdown the thread pool."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def effective_n_jobs(self, n_jobs: int) -> int:
        """Return the effective number of jobs."""
        if n_jobs == 0:
            return 0
        if n_jobs == 1:
            return 1
        return self.pool.total_slots

    def submit(
        self,
        func: Callable[[], Any],
        callback: Callable[[Future[Any]], None] | None = None,
    ) -> Future[Any]:
        """Submit a task for async execution."""
        if self._executor is None:
            raise RuntimeError("Backend not configured")

        logger.debug("Submitting task to Skyward backend")
        ctx = contextvars.copy_context()
        future = self._executor.submit(ctx.run, self._execute_on_skyward, func)
        if callback is not None:
            future.add_done_callback(callback)
        return future

    def _execute_on_skyward(self, func: Callable[[], Any]) -> Any:
        """Execute a function on Skyward (blocking, runs in thread)."""
        logger.debug("Executing task on Skyward pool")
        payload = cloudpickle.dumps(func)
        pending = self._remote_execute(payload)
        result = pending >> self.pool
        logger.debug("Task execution completed")
        return result

    def retrieve_result_callback(self, future: Future[Any]) -> Any:
        """Extract result from future (called from callback)."""
        return future.result()

    def get_nested_backend(self) -> tuple[ParallelBackendBase, int]:
        """Return backend for nested parallelism."""
        from joblib._parallel_backends import SequentialBackend

        return SequentialBackend(nesting_level=self.nesting_level + 1), None

    def abort_everything(self, ensure_ready: bool = True) -> None:
        """Abort all pending tasks."""
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            if ensure_ready:
                self._executor = ThreadPoolExecutor(
                    max_workers=self.effective_n_jobs(self._n_jobs)
                )


_backend_registered = False


def _setup_backend(pool: ComputePool) -> None:
    """Register the Skyward backend for joblib."""
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

    Args:
        pool: The ComputePool to use for distributed execution.

    Example:
        with ComputePool(provider=AWS(), nodes=4) as pool:
            with sklearn_backend(pool):
                grid_search = GridSearchCV(estimator, param_grid, n_jobs=-1)
                grid_search.fit(X, y)  # Distributed!
    """
    logger.info("Activating sklearn/joblib backend with Skyward pool")
    _setup_backend(pool)
    with parallel_backend("skyward"):
        yield
    logger.debug("sklearn/joblib backend deactivated")


joblib_backend = sklearn_backend


def _merge_pip(base: Image, *packages: str) -> Image:
    """Create new Image with additional pip packages."""
    return Image(
        python=base.python,
        pip=[*base.pip, *packages],
        pip_extra_index_url=base.pip_extra_index_url,
        apt=base.apt,
        env=base.env,
    )


@contextmanager
def JoblibPool(
    provider: Provider,
    *,
    nodes: int = 1,
    machine: str | None = None,
    image: Image | None = None,
    accelerator: Accelerator | str | None = None,
    cpu: int | None = None,
    memory: Memory | None = None,
    volume: dict[str, str] | Sequence[Volume] | None = None,
    allocation: AllocationLike = "spot-if-available",
    timeout: int = 3600,
    env: dict[str, str] | None = None,
    concurrency: int = 1,
    display: Literal["spinner", "quiet"] = "spinner",
    on_event: Callback | None = None,
    collect_metrics: bool = True,
    joblib_version: str | None = None,
) -> Iterator[ComputePool]:
    """Compute pool with joblib backend for distributed parallel execution.

    Automatically adds joblib to pip dependencies and sets up the backend.

    Args:
        provider: Cloud provider (AWS, DigitalOcean, Verda).
        nodes: Number of nodes to provision.
        machine: Direct instance type override (e.g., "p5.48xlarge").
        image: Base image. joblib added automatically.
        accelerator: GPU/accelerator specification.
        cpu: CPU cores per worker.
        memory: Memory per worker (e.g., "32GB").
        volume: Volumes to mount.
        allocation: Instance allocation strategy.
        timeout: Task timeout in seconds.
        env: Environment variables.
        concurrency: Concurrent tasks per node.
        display: Output display mode.
        on_event: Event callback.
        collect_metrics: Whether to collect metrics.
        joblib_version: Specific joblib version (e.g., "1.3.0"). None for latest.

    Yields:
        The active ComputePool.

    Example:
        with JoblibPool(provider=AWS(), nodes=4, concurrency=4) as pool:
            results = Parallel(n_jobs=-1)(delayed(fn)(x) for x in data)
    """
    logger.info(f"Creating JoblibPool with {nodes} nodes, concurrency={concurrency}")
    base = image or DEFAULT_IMAGE
    pkg = f"joblib=={joblib_version}" if joblib_version else "joblib"
    merged = _merge_pip(base, pkg)

    pool = ComputePool(
        provider=provider,
        nodes=nodes,
        machine=machine,
        image=merged,
        accelerator=accelerator,
        cpu=cpu,
        memory=memory,
        volume=volume,
        allocation=allocation,
        timeout=timeout,
        env=env,
        concurrency=concurrency,
        display=display,
        on_event=on_event,
        collect_metrics=collect_metrics,
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
    machine: str | None = None,
    image: Image | None = None,
    accelerator: Accelerator | str | None = None,
    cpu: int | None = None,
    memory: Memory | None = None,
    volume: dict[str, str] | Sequence[Volume] | None = None,
    allocation: AllocationLike = "spot-if-available",
    timeout: int = 3600,
    env: dict[str, str] | None = None,
    concurrency: int = 1,
    display: Literal["spinner", "quiet"] = "spinner",
    on_event: Callback | None = None,
    collect_metrics: bool = True,
    sklearn_version: str | None = None,
) -> Iterator[ComputePool]:
    """Compute pool with scikit-learn for distributed ML training.

    Automatically adds scikit-learn to pip dependencies and sets up the backend.

    Args:
        provider: Cloud provider (AWS, DigitalOcean, Verda).
        nodes: Number of nodes to provision.
        machine: Direct instance type override (e.g., "p5.48xlarge").
        image: Base image. scikit-learn added automatically.
        accelerator: GPU/accelerator specification.
        cpu: CPU cores per worker.
        memory: Memory per worker (e.g., "32GB").
        volume: Volumes to mount.
        allocation: Instance allocation strategy.
        timeout: Task timeout in seconds.
        env: Environment variables.
        concurrency: Concurrent tasks per node.
        display: Output display mode.
        on_event: Event callback.
        collect_metrics: Whether to collect metrics.
        sklearn_version: Specific sklearn version (e.g., "1.4.0"). None for latest.

    Yields:
        The active ComputePool.

    Example:
        with ScikitLearnPool(provider=AWS(), nodes=4) as pool:
            grid = GridSearchCV(model, params, n_jobs=-1)
            grid.fit(X, y)  # Distributed!
    """
    logger.info(f"Creating ScikitLearnPool with {nodes} nodes, concurrency={concurrency}")
    base = image or DEFAULT_IMAGE
    pkg = f"scikit-learn=={sklearn_version}" if sklearn_version else "scikit-learn"
    merged = _merge_pip(base, pkg)

    pool = ComputePool(
        provider=provider,
        nodes=nodes,
        machine=machine,
        image=merged,
        accelerator=accelerator,
        cpu=cpu,
        memory=memory,
        volume=volume,
        allocation=allocation,
        timeout=timeout,
        env=env,
        concurrency=concurrency,
        display=display,
        on_event=on_event,
        collect_metrics=collect_metrics,
    )

    with pool:
        _setup_backend(pool)
        with parallel_backend("skyward"):
            yield pool
