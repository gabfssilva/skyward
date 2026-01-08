"""Skyward Executor - concurrent.futures.Executor with auto-provisioning.

Provides a drop-in replacement for ThreadPoolExecutor that runs tasks
on provisioned cloud instances.

    from skyward import SkywardExecutor, AWS

    with SkywardExecutor(provider=AWS(), nodes=4, concurrency=4) as executor:
        futures = [executor.submit(process, item) for item in items]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # Or with map:
        results = list(executor.map(process, items))
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from contextlib import suppress
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal

import cloudpickle
from loguru import logger

from skyward.core.callback import Callback
from skyward.observability.logging import LogConfig
from skyward.pool.compute import ComputePool
from skyward.spec.allocation import AllocationLike
from skyward.spec.image import DEFAULT_IMAGE, Image
from skyward.spec.volume import Volume

if TYPE_CHECKING:
    from skyward.types import AcceleratorSpec, ProviderLike

from skyward.types import Memory


def _make_remote_executor() -> Callable[[bytes, bytes, bytes], Any]:
    """Create the remote execution function for arbitrary callables."""
    from skyward import compute

    @compute
    def execute_remote(fn_bytes: bytes, args_bytes: bytes, kwargs_bytes: bytes) -> Any:
        """Execute serialized function on remote instance."""
        import cloudpickle

        fn = cloudpickle.loads(fn_bytes)
        args = cloudpickle.loads(args_bytes)
        kwargs = cloudpickle.loads(kwargs_bytes)
        return fn(*args, **kwargs)

    return execute_remote


class Executor(Executor):
    """Executor that provisions cloud instances for task execution.

    Implements the concurrent.futures.Executor interface, allowing drop-in
    replacement for ThreadPoolExecutor while running tasks on cloud machines.

    The executor provisions a ComputePool on __enter__ and shuts it down on
    __exit__. Tasks are submitted to the pool and executed remotely.

    Args:
        provider: Cloud provider (AWS, DigitalOcean, Verda).
        nodes: Number of nodes to provision.
        machine: Direct instance type override (e.g., "p5.48xlarge").
        image: Base image specification.
        accelerator: GPU/accelerator specification.
        cpu: CPU cores per worker.
        memory: Memory per worker (e.g., "32GB").
        volume: Volumes to mount.
        allocation: Instance allocation strategy.
        timeout: Task timeout in seconds.
        env: Environment variables.
        max_hourly_cost: Maximum USD/hour for entire cluster.
        concurrency: Concurrent tasks per node.
        display: Output display mode.
        on_event: Event callback.

    Example:
        with SkywardExecutor(provider=AWS(), nodes=4, concurrency=4) as executor:
            # Submit individual tasks
            future = executor.submit(process, item)
            result = future.result()

            # Or map over items
            results = list(executor.map(process, items))

            # Works with as_completed
            futures = [executor.submit(fn, x) for x in data]
            for future in concurrent.futures.as_completed(futures):
                print(future.result())
    """

    def __init__(
        self,
        provider: ProviderLike,
        *,
        nodes: int = 1,
        machine: str | None = None,
        image: Image | None = None,
        accelerator: AcceleratorSpec | str | None = None,
        cpu: int | None = None,
        memory: Memory | None = None,
        volume: dict[str, str] | Sequence[Volume] | None = None,
        allocation: AllocationLike = "spot-if-available",
        timeout: int = 3600,
        env: dict[str, str] | None = None,
        max_hourly_cost: float | None = None,
        concurrency: int = 1,
        display: Literal["panel", "spinner", "quiet"] = "panel",
        on_event: Callback | None = None,
        logging: LogConfig | bool = True,
    ) -> None:
        self._provider = provider
        self._nodes = nodes
        self._machine = machine
        self._image = image or DEFAULT_IMAGE
        self._accelerator = accelerator
        self._cpu = cpu
        self._memory = memory
        self._volume = volume
        self._allocation = allocation
        self._timeout = timeout
        self._env = env
        self._max_hourly_cost = max_hourly_cost
        self._concurrency = concurrency
        self._display = display
        self._on_event = on_event
        self._logging = logging

        self._pool: ComputePool | None = None
        self._thread_executor: ThreadPoolExecutor | None = None
        self._remote_execute = _make_remote_executor()
        self._shutdown = False

    def __enter__(self) -> Executor:
        """Enter context and provision cloud resources."""
        logger.debug(f"Initializing executor with {self._nodes} nodes, concurrency={self._concurrency}")

        self._pool = ComputePool(
            provider=self._provider,
            nodes=self._nodes,
            machine=self._machine,
            image=self._image,
            accelerator=self._accelerator,
            cpu=self._cpu,
            memory=self._memory,
            volume=self._volume,
            allocation=self._allocation,
            timeout=self._timeout,
            env=self._env,
            max_hourly_cost=self._max_hourly_cost,
            concurrency=self._concurrency,
            display=self._display,
            on_event=self._on_event,
            logging=self._logging,
        )
        self._pool.__enter__()

        # Thread executor for async future management
        # Use total_slots as max_workers to match pool capacity
        self._thread_executor = ThreadPoolExecutor(max_workers=self._pool.total_slots)
        logger.debug(f"Executor ready with {self._pool.total_slots} execution slots")

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context and release resources."""
        self.shutdown(wait=True)

    @property
    def total_slots(self) -> int:
        """Total number of execution slots (nodes * concurrency)."""
        if self._pool is None:
            return self._nodes * self._concurrency
        return self._pool.total_slots

    def submit[T](
        self,
        fn: Callable[..., T],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Future[T]:
        """Submit a callable for execution on the cloud pool.

        Args:
            fn: Callable to execute.
            *args: Positional arguments for the callable.
            **kwargs: Keyword arguments for the callable.

        Returns:
            Future representing pending execution.

        Raises:
            RuntimeError: If executor is not active or already shut down.
        """
        if self._shutdown:
            raise RuntimeError("Cannot submit to a shut down executor")
        if self._pool is None or self._thread_executor is None:
            raise RuntimeError("Executor not active. Use within context manager.")

        logger.debug(f"Submitting task: {fn.__name__}")
        # Propagate contextvars to worker thread
        ctx = contextvars.copy_context()
        return self._thread_executor.submit(ctx.run, self._execute_on_pool, fn, args, kwargs)

    def _execute_on_pool(
        self,
        fn: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Execute function on Skyward pool (blocking, runs in thread)."""
        assert self._pool is not None

        # Serialize function and arguments
        fn_bytes = cloudpickle.dumps(fn)
        args_bytes = cloudpickle.dumps(args)
        kwargs_bytes = cloudpickle.dumps(kwargs)

        # Create pending compute and execute
        pending = self._remote_execute(fn_bytes, args_bytes, kwargs_bytes)
        return pending >> self._pool

    def map[T](
        self,
        fn: Callable[..., T],
        *iterables: Iterable[Any],
        timeout: float | None = None,
        chunksize: int = 1,
    ) -> Iterator[T]:
        """Map function over iterables, executing on cloud pool.

        Args:
            fn: Callable to map.
            *iterables: Iterables to zip and map over.
            timeout: Maximum time to wait for each result.
            chunksize: Not used (exists for API compatibility).

        Yields:
            Results in order of input.

        Raises:
            TimeoutError: If a result is not available within timeout.
        """
        del chunksize  # Unused, exists for API compatibility

        # Submit all tasks
        futures: list[Future[T]] = []
        for item_args in zip(*iterables, strict=False):
            if len(item_args) == 1:
                futures.append(self.submit(fn, item_args[0]))
            else:
                futures.append(self.submit(fn, *item_args))

        # Yield results in order
        for future in futures:
            yield future.result(timeout=timeout)

    def shutdown(
        self,
        wait: bool = True,
        *,
        cancel_futures: bool = False,
    ) -> None:
        """Shutdown the executor and release cloud resources.

        Args:
            wait: If True, wait for pending futures to complete.
            cancel_futures: If True, cancel all pending futures.
        """
        if self._shutdown:
            return

        logger.debug(f"Shutting down executor (wait={wait}, cancel_futures={cancel_futures})")
        self._shutdown = True

        # Shutdown thread executor first
        if self._thread_executor is not None:
            logger.debug("Shutting down thread executor...")
            self._thread_executor.shutdown(wait=wait, cancel_futures=cancel_futures)
            self._thread_executor = None

        # Then shutdown cloud pool
        if self._pool is not None:
            logger.debug("Shutting down cloud pool...")
            with suppress(Exception):
                self._pool.__exit__(None, None, None)
            self._pool = None

        logger.debug("Executor shutdown complete")
