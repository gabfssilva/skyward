"""Skyward joblib backend for sklearn integration.

Enables distributed sklearn training with zero code changes:

    from skyward import ComputePool, AWS, Image
    from skyward.integrations import sklearn_backend

    with ComputePool(
        provider=AWS(),
        nodes=4,
        image=Image(pip=["scikit-learn"]),
    ) as pool:
        with sklearn_backend(pool):
            # Any sklearn code with n_jobs=-1 is now distributed!
            grid_search = GridSearchCV(estimator, param_grid, n_jobs=-1)
            grid_search.fit(X, y)

Uses joblib's pluggable backend system for seamless sklearn integration.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import cloudpickle
from joblib import parallel_backend
from joblib.parallel import ParallelBackendBase, register_parallel_backend

if TYPE_CHECKING:
    from joblib import Parallel

    from skyward.pool import ComputePool


def _make_remote_executor() -> Callable[[bytes], Any]:
    """Create the remote execution function."""
    from skyward import compute

    @compute
    def execute_batch(payload: bytes) -> Any:
        """Execute a serialized BatchedCalls on a remote worker."""
        import cloudpickle

        print("running...")
        batch_func = cloudpickle.loads(payload)
        result = batch_func()
        print("done!")
        return result

    return execute_batch


_remote_execute = _make_remote_executor()


class SkywardBackend(ParallelBackendBase):
    """Joblib backend that dispatches tasks to Skyward workers.

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
        # Create a thread pool for async submission to Skyward
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
        # Use 2x nodes for good parallelism (some tasks may be waiting on I/O)
        return max(self.pool.nodes * 2, 4)

    def submit(
        self,
        func: Callable[[], Any],
        callback: Callable[[Future[Any]], None] | None = None,
    ) -> Future[Any]:
        """Submit a task for async execution.

        Args:
            func: The callable to execute (BatchedCalls from joblib).
            callback: Called when the task completes.

        Returns:
            A Future that will contain the result.
        """
        if self._executor is None:
            raise RuntimeError("Backend not configured")

        # Copy context to propagate callback contextvars to worker threads
        ctx = contextvars.copy_context()
        future = self._executor.submit(ctx.run, self._execute_on_skyward, func)
        if callback is not None:
            future.add_done_callback(callback)
        return future

    def _execute_on_skyward(self, func: Callable[[], Any]) -> Any:
        """Execute a function on Skyward (blocking, runs in thread)."""
        # Serialize the batch function
        payload = cloudpickle.dumps(func)

        # Execute via Skyward pool (this blocks, but we're in a thread)
        pending = _remote_execute(payload)
        return pending >> self.pool

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
    global _backend_registered

    def backend_factory() -> SkywardBackend:
        return SkywardBackend(pool)

    if not _backend_registered:
        register_parallel_backend("skyward", backend_factory)
        _backend_registered = True
    else:
        register_parallel_backend("skyward", backend_factory, make_default=False)

    with parallel_backend("skyward"):
        yield


joblib_backend = sklearn_backend
