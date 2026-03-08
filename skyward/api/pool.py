"""Synchronous facade for v2 async pool.

This module provides the user-facing synchronous API that mirrors v1:

    import skyward as sky

    @sky.function
    def train(data):
        return model.fit(data)

    with sky.ComputePool(provider=sky.AWS(), accelerator=sky.accelerators.A100(), nodes=4) as pool:
        result = train(data) >> pool         # execute on one node
        results = train(data) @ pool         # broadcast to all nodes
        a, b = (task1() & task2()) >> pool   # parallel execution

Internally, this facade:
1. Runs v2's async provisioning in a sync context
2. Manages the asyncio event loop lifecycle
3. Provides >> and @ operator support via the sky singleton
"""

from __future__ import annotations

import asyncio
import functools
import queue
import threading
import types
from collections.abc import Callable, Coroutine, Generator, Iterator, Sequence
from concurrent.futures import Future
from contextlib import suppress
from contextvars import ContextVar, Token
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, overload

from casty import ActorRef, ActorSystem, Behaviors, CastyConfig

from skyward.accelerators import Accelerator
from skyward.actors.messages import (
    CurrentNodeCount,
    GetCurrentNodes,
    NodeInstance,
    PoolMsg,
    ProvisionFailed,
    SubmitBroadcast,
    SubmitTask,
    TaskResult,
)
from skyward.api.provider import ProviderConfig
from skyward.distributed import (
    BarrierProxy,
    CounterProxy,
    DictProxy,
    DistributedRegistry,
    LockProxy,
    QueueProxy,
    SetProxy,
    _set_active_registry,
)
from skyward.distributed.types import Consistency
from skyward.observability.logger import logger
from skyward.observability.logging import LogConfig, setup_logging, teardown_logging
from skyward.plugins.plugin import Plugin

from .model import Offer
from .spec import DEFAULT_IMAGE, Image, PoolSpec, SelectionStrategy, Spec, Volume, Worker

if TYPE_CHECKING:
    from skyward.api.model import Cluster

_active_pool: ContextVar[ComputePool | None] = ContextVar("active_pool", default=None)


def _cancel_pending_tasks() -> None:
    current = asyncio.current_task()
    for task in asyncio.all_tasks():
        if task is not current and not task.done():
            task.cancel()


def _get_active_pool() -> ComputePool:
    """Get the active pool from context."""
    pool = _active_pool.get()
    if pool is None:
        raise RuntimeError(
            "No active pool. Use within a @pool decorated function or 'with pool(...):' block."
        )
    return pool


class _Sky:
    """Singleton that captures >> and @ operators.

    This allows the v1-style API:
        result = compute_fn(args) >> sky   # execute on one node
        results = compute_fn(args) @ sky   # broadcast to all nodes
    """

    def __rrshift__(self, pending: PendingFunction[Any] | PendingFunctionGroup) -> Any:
        """pending >> sky - execute computation(s)."""
        pool = _get_active_pool()
        match pending:
            case PendingFunctionGroup():
                return pool.run_parallel(pending)
            case _:
                return pool.run(pending)

    def __rmatmul__(self, pending: PendingFunction[Any]) -> list[Any]:
        """pending @ sky - broadcast to all nodes."""
        pool = _get_active_pool()
        return pool.broadcast(pending)

    def __repr__(self) -> str:
        pool = _active_pool.get()
        if pool:
            return f"<sky: active pool with {pool._specs[0].nodes} nodes>"
        return "<sky: no active pool>"


sky = _Sky()


@dataclass(frozen=True, slots=True)
class PendingFunction[T]:
    """Lazy computation wrapper.

    Represents a function call that will be executed remotely
    when sent to a pool via the >> or @ operator.

    Example:
        @compute
        def train(data):
            return model.fit(data)

        pending = train(data)  # Returns PendingCompute, doesn't execute
        result = pending >> sky  # Executes remotely on pool
        results = pending @ sky  # Broadcasts to all nodes

        # Override timeout
        result = train(data).with_timeout(600) >> sky
    """

    fn: Callable[..., T]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    timeout: float | None = None

    def with_timeout(self, timeout: float) -> PendingFunction[T]:
        return PendingFunction(fn=self.fn, args=self.args, kwargs=self.kwargs, timeout=timeout)

    def __rshift__(self, target: ComputePool | _Sky | types.ModuleType) -> T:
        match target:
            case types.ModuleType() if hasattr(target, "sky"):
                return target.sky.__rrshift__(self)  # type: ignore
            case _Sky():
                return target.__rrshift__(self)  # type: ignore
            case _:
                return target.run(self)  # type: ignore[union-attr]

    def __gt__(self, target: ComputePool | _Sky | types.ModuleType) -> Future[T]:
        match target:
            case types.ModuleType() if hasattr(target, "sky"):
                return target.sky._run_async(self)  # type: ignore
            case _Sky():
                return target._run_async(self)  # type: ignore
            case _:
                return target.run_async(self)  # type: ignore[union-attr]

    def __matmul__(self, target: ComputePool | _Sky | types.ModuleType) -> list[T] | tuple[T, ...]:
        """Broadcast to all nodes using @ operator."""
        match target:
            case types.ModuleType() if hasattr(target, "sky"):
                return target.sky.__rmatmul__(self)
            case _Sky():
                return target.__rmatmul__(self)
            case _:
                return target.broadcast(self)  # type: ignore[union-attr]

    def __and__(self, other: PendingFunction[Any] | PendingFunctionGroup) -> PendingFunctionGroup:
        """Combine with another computation for parallel execution."""
        match other:
            case PendingFunctionGroup():
                return PendingFunctionGroup(items=(self, *other.items))
            case _:
                return PendingFunctionGroup(items=(self, other))


@dataclass(frozen=True, slots=True)
class PendingFunctionGroup:
    """Group of computations for parallel execution.

    Created by using the & operator:
        group = task1() & task2() & task3()
        a, b, c = group >> sky

    Or using gather():
        group = gather(task1(), task2(), task3())
        results = group >> sky
    """

    items: tuple[PendingFunction[Any], ...]
    stream: bool = False
    ordered: bool = True
    timeout: float | None = None

    def with_timeout(self, timeout: float) -> PendingFunctionGroup:
        return PendingFunctionGroup(
            items=self.items, stream=self.stream,
            ordered=self.ordered, timeout=timeout,
        )

    def __and__(self, other: PendingFunction[Any] | PendingFunctionGroup) -> PendingFunctionGroup:
        """Add another computation to the group."""
        match other:
            case PendingFunctionGroup():
                return PendingFunctionGroup(
                    items=(*self.items, *other.items),
                    stream=self.stream, ordered=self.ordered,
                )
            case _:
                return PendingFunctionGroup(
                    items=(*self.items, other),
                    stream=self.stream, ordered=self.ordered,
                )

    def __rshift__(
        self, target: ComputePool | _Sky | types.ModuleType
    ) -> tuple[Any, ...] | Generator[Any, None, None]:
        """Execute all computations in parallel using >> operator."""
        match target:
            case types.ModuleType() if hasattr(target, "sky"):
                return target.sky.__rrshift__(self)  # type: ignore
            case _Sky():
                return target.__rrshift__(self)  # type: ignore
            case _:
                return target.run_parallel(self)  # type: ignore[union-attr]

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[PendingFunction[Any]]:
        return iter(self.items)


def gather(
    *pendings: PendingFunction[Any],
    stream: bool = False,
    ordered: bool = True,
) -> PendingFunctionGroup:
    """Group computations for parallel execution.

    Example:
        results = gather(task1(), task2(), task3()) >> sky
        # results is a tuple of (result1, result2, result3)

        for result in gather(task1(), task2(), task3(), stream=True) >> sky:
            print(result)  # yields results as they complete

        for result in gather(task1(), task2(), task3(), stream=True, ordered=False) >> sky:
            print(result)  # yields results as they complete, unordered
    """
    return PendingFunctionGroup(items=pendings, stream=stream, ordered=ordered)


@overload
def function[**P, T](fn: Callable[P, T]) -> Callable[P, PendingFunction[T]]: ...

@overload
def function[**P, T](
    *, timeout: float,
) -> Callable[[Callable[P, T]], Callable[P, PendingFunction[T]]]: ...

def function[**P, T](
    fn: Callable[P, T] | None = None,
    *,
    timeout: float | None = None,
) -> Callable[P, PendingFunction[T]] | Callable[[Callable[P, T]], Callable[P, PendingFunction[T]]]:
    def decorator(f: Callable[P, T]) -> Callable[P, PendingFunction[T]]:
        @functools.wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> PendingFunction[T]:
            return PendingFunction(fn=f, args=args, kwargs=kwargs, timeout=timeout)
        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator


_FDS_PER_NODE: int = 10
_FD_BASE_OVERHEAD: int = 50


def _check_fd_budget(nodes: int) -> None:
    import resource

    estimated = nodes * _FDS_PER_NODE + _FD_BASE_OVERHEAD
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft >= estimated:
        return

    target = min(int(estimated * 1.5), hard)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        logger.info(
            "Raised file descriptor limit from {old} to {new}",
            old=soft, new=target,
        )
    except (ValueError, OSError):
        logger.warning(
            "File descriptor limit ({soft}) may be insufficient for {nodes} nodes "
            "(estimated need: {estimated}). Consider running: ulimit -n {target}",
            soft=soft, nodes=nodes, estimated=estimated, target=target,
        )


class ComputePool:
    """Provision cloud compute and execute functions remotely.

    Two usage modes:

        # Single provider (legacy)
        with ComputePool(provider=AWS(), accelerator=A100(), nodes=4) as pool:
            result = train(data) >> pool

        # Multi-spec with fallback
        with ComputePool(
            Spec(provider=VastAI(), accelerator=A100()),
            Spec(provider=AWS(), accelerator=A100()),
            selection="cheapest",
        ) as pool:
            result = train(data) >> pool
    """

    @overload
    def __init__(
        self, *,
        provider: ProviderConfig,
        nodes: int | tuple[int, int] = ...,
        accelerator: Accelerator | None = ...,
        vcpus: float | None = ...,
        memory_gb: float | None = ...,
        architecture: Literal["x86_64", "arm64"] | None = ...,
        allocation: Literal["spot", "on-demand", "spot-if-available"] = ...,
        image: Image = ...,
        ttl: int = ...,
        worker: Worker | None = ...,
        logging: LogConfig | bool = ...,
        max_hourly_cost: float | None = ...,
        default_compute_timeout: float = ...,
        provision_timeout: int = ...,
        ssh_timeout: int = ...,
        ssh_retry_interval: int = ...,
        provision_retry_delay: float = ...,
        max_provision_attempts: int = ...,
        volumes: list[Volume] | tuple[Volume, ...] = ...,
        autoscale_cooldown: float = ...,
        autoscale_idle_timeout: float = ...,
        reconcile_tick_interval: float = ...,
        plugins: list[Plugin] | tuple[Plugin, ...] = ...,
        shutdown_timeout: float = ...,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *specs: Spec,
        selection: SelectionStrategy = ...,
        image: Image = ...,
        worker: Worker | None = ...,
        logging: LogConfig | bool = ...,
        default_compute_timeout: float = ...,
        provision_timeout: int = ...,
        ssh_timeout: int = ...,
        ssh_retry_interval: int = ...,
        provision_retry_delay: float = ...,
        max_provision_attempts: int = ...,
        volumes: list[Volume] | tuple[Volume, ...] = ...,
        autoscale_cooldown: float = ...,
        autoscale_idle_timeout: float = ...,
        reconcile_tick_interval: float = ...,
        plugins: list[Plugin] | tuple[Plugin, ...] = ...,
        shutdown_timeout: float = ...,
    ) -> None: ...

    def __init__(
        self,
        *specs: Spec,
        provider: ProviderConfig | None = None,
        nodes: int | tuple[int, int] = 1,
        accelerator: Accelerator | None = None,
        vcpus: float | None = None,
        memory_gb: float | None = None,
        architecture: Literal["x86_64", "arm64"] | None = None,
        allocation: Literal["spot", "on-demand", "spot-if-available"] = "spot-if-available",
        selection: SelectionStrategy = "cheapest",
        image: Image = DEFAULT_IMAGE,
        ttl: int = 600,
        worker: Worker | None = None,
        logging: LogConfig | bool = True,
        max_hourly_cost: float | None = None,
        default_compute_timeout: float = 300.0,
        provision_timeout: int = 300,
        ssh_timeout: int = 300,
        ssh_retry_interval: int = 2,
        provision_retry_delay: float = 5.0,
        max_provision_attempts: int = 3,
        volumes: list[Volume] | tuple[Volume, ...] = (),
        autoscale_cooldown: float = 30.0,
        autoscale_idle_timeout: float = 60.0,
        reconcile_tick_interval: float = 15.0,
        plugins: list[Plugin] | tuple[Plugin, ...] = (),
        shutdown_timeout: float = 120.0,
    ) -> None:
        if specs and provider is not None:
            raise ValueError("Cannot specify both positional Spec args and 'provider'")
        if not specs and provider is None:
            raise ValueError("Either Spec args or 'provider' must be provided")

        if specs:
            self._specs = specs
            self._scaling: tuple[int, int] | None = None
        else:
            assert provider is not None
            match nodes:
                case (min_n, max_n):
                    self._scaling = (min_n, max_n)
                case _:
                    self._scaling = None
            self._specs = (Spec(
                provider=provider,
                accelerator=accelerator,
                nodes=nodes,
                vcpus=vcpus,
                memory_gb=memory_gb,
                architecture=architecture,
                allocation=allocation,
                region=getattr(provider, "region", None),
                max_hourly_cost=max_hourly_cost,
                ttl=ttl,
            ),)

        self.selection = selection
        self.image = image
        self.worker = worker or Worker()
        self.logging = logging
        self.default_compute_timeout = default_compute_timeout
        self.provision_timeout = provision_timeout
        self.ssh_timeout = ssh_timeout
        self.ssh_retry_interval = ssh_retry_interval
        self.provision_retry_delay = provision_retry_delay
        self.max_provision_attempts = max_provision_attempts
        self.volumes = tuple(volumes)
        self._plugins = tuple(plugins)
        self.autoscale_cooldown = autoscale_cooldown
        self.autoscale_idle_timeout = autoscale_idle_timeout
        self.reconcile_tick_interval = reconcile_tick_interval
        self.shutdown_timeout = shutdown_timeout

        self._log_handler_ids: list[int] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._active: bool = False
        self._context_token: Token[ComputePool | None] | None = None
        self._registry: DistributedRegistry | None = None
        self._system: ActorSystem | None = None
        self._pool_ref: ActorRef[PoolMsg] | None = None
        self._cluster_id: str = ""
        self._cluster: Cluster[Any] | None = None
        self._instances: dict[int, NodeInstance] = {}
        self._spec: PoolSpec | None = None
        self._plugin_client_contexts: list[Any] = []
        self._app: Any = None
        self._owns_app: bool = False

    def _build_specs(self) -> list[Spec]:
        return list(self._specs)

    def _apply_plugin_transforms(self, image: Image, cluster: Cluster[Any]) -> Image:
        """Apply plugin Image transforms sequentially."""
        for plugin in self._plugins:
            if plugin.transform is not None:
                image = plugin.transform(image, cluster)
        return image

    def _collect_plugin_bootstrap(self, cluster: Cluster[Any]) -> tuple:
        """Collect bootstrap ops from all plugins."""
        ops: list[Any] = []
        for plugin in self._plugins:
            if plugin.bootstrap is not None:
                ops.extend(plugin.bootstrap(cluster))
        return tuple(ops)

    def _decorate_fn(self, fn: Any) -> Any:
        """Wrap fn with plugin decorate chains."""
        from skyward.plugins.plugin import chain_decorators

        decorators: list[Any] = [p.decorate for p in self._plugins if p.decorate is not None]
        return chain_decorators(fn, decorators)

    def __enter__(self) -> ComputePool:
        """Start pool and provision resources."""
        if self.logging:
            match self.logging:
                case True:
                    log_config = LogConfig(console=False)
                case _:
                    log_config = LogConfig(
                        level=self.logging.level,
                        file=self.logging.file,
                        console=False,
                        rotation=self.logging.rotation,
                        retention=self.logging.retention,
                    )
            self._log_handler_ids = setup_logging(log_config)

        first = self._specs[0]
        match first.nodes:
            case (min_n, max_n):
                logger.info(
                    "Starting pool with {min}-{max} nodes ({accel})",
                    min=min_n, max=max_n, accel=first.accelerator,
                )
                fd_nodes = max_n
            case int(n):
                logger.info(
                    "Starting pool with {n} nodes ({accel})",
                    n=n, accel=first.accelerator,
                )
                fd_nodes = n

        from skyward.app import App, get_app

        app = get_app()
        if app is None:
            app = App()
            app.__enter__()
            self._owns_app = True
        self._app = app

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="skyward-event-loop",
        )
        self._loop_thread.start()

        _check_fd_budget(fd_nodes)

        try:
            self._run_sync(self._start_async())
            self._active = True
            self._context_token = _active_pool.set(self)

            # Enter plugin around_client contexts
            assert self._cluster is not None
            for plugin in self._plugins:
                if plugin.around_client is not None:
                    ctx = plugin.around_client(self, self._cluster)
                    ctx.__enter__()
                    self._plugin_client_contexts.append(ctx)

            logger.info("Pool ready")
        except Exception as e:
            logger.exception("Error starting pool: {err}", err=e)
            self._cleanup()
            raise

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Stop pool and release resources."""
        # Exit plugin around_client contexts (reverse order)
        for ctx in reversed(self._plugin_client_contexts):
            try:
                ctx.__exit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                logger.warning("Plugin around_client exit error: {err}", err=e)
        self._plugin_client_contexts.clear()

        logger.info("Stopping pool...")
        logger.debug(
            "ComputePool.__exit__: _active={active}, _pool_ref={ref}, _system={sys}",
            active=self._active,
            ref=self._pool_ref is not None,
            sys=self._system is not None,
        )
        try:
            if self._context_token is not None:
                _active_pool.reset(self._context_token)
                self._context_token = None

            if self._active:
                self._run_sync_with_timeout(self._stop_async(), timeout=self.shutdown_timeout)
        except TimeoutError:
            logger.warning(
                "Pool stop timed out after {t}s, forcing cleanup",
                t=self.shutdown_timeout,
            )
        except Exception as e:
            logger.warning("Error stopping pool: {err}", err=e)
        except BaseException as e:
            logger.error("Fatal error stopping pool: {err}", err=e)
        finally:
            if self._registry is not None:
                self._registry.cleanup()
                _set_active_registry(None)
                self._registry = None

            self._active = False

            if self._owns_app and self._app is not None:
                self._app.__exit__(None, None, None)
                self._app = None
                self._owns_app = False

            self._cleanup()
            logger.info("Pool stopped")

            if self._log_handler_ids:
                teardown_logging(self._log_handler_ids)

    @staticmethod
    def _unwrap_broadcast_result(value: Any) -> Any:
        match value:
            case RuntimeError() as err:
                raise err
            case _:
                return value

    def _unwrap_result(self, result: TaskResult) -> Any:
        return self._unwrap_broadcast_result(result.value)

    def _resolve_timeout(self, pending: PendingFunction[Any]) -> float:
        return pending.timeout if pending.timeout is not None else self.default_compute_timeout

    def _submit(self, pending: PendingFunction[Any]) -> Callable[[ActorRef[Any]], SubmitTask]:
        timeout = self._resolve_timeout(pending)
        fn = self._decorate_fn(pending.fn)
        return lambda reply_to: SubmitTask(
            fn=fn, args=pending.args, kwargs=pending.kwargs,
            reply_to=reply_to, timeout=timeout,
        )

    def run[T](self, pending: PendingFunction[T]) -> T:
        if not self._active or self._pool_ref is None or self._system is None:
            raise RuntimeError("Pool is not active")

        timeout = self._resolve_timeout(pending)
        logger.debug("Submitting task: {fn}", fn=getattr(pending.fn, "__name__", repr(pending.fn)))
        result: TaskResult = self._run_sync(
            self._system.ask(self._pool_ref, self._submit(pending), timeout=timeout)
        )
        return self._unwrap_result(result)

    def run_async[T](self, pending: PendingFunction[T]) -> Future[T]:
        if not self._active or self._pool_ref is None or self._system is None or self._loop is None:
            raise RuntimeError("Pool is not active")

        timeout = self._resolve_timeout(pending)
        fn_name = getattr(pending.fn, "__name__", repr(pending.fn))
        logger.debug("Submitting async task: {fn}", fn=fn_name)

        async def _run() -> T:
            assert self._pool_ref is not None
            result: TaskResult = await self._system.ask(  # type: ignore[union-attr]
                self._pool_ref, self._submit(pending), timeout=timeout,
            )
            return self._unwrap_result(result)

        return asyncio.run_coroutine_threadsafe(_run(), self._loop)

    def broadcast[T](self, pending: PendingFunction[T]) -> list[T]:
        if not self._active or self._pool_ref is None or self._system is None:
            raise RuntimeError("Pool is not active")

        timeout = self._resolve_timeout(pending)
        fn = self._decorate_fn(pending.fn)
        fn_name = getattr(pending.fn, "__name__", repr(pending.fn))
        logger.debug("Broadcasting task: {fn} to {n} nodes", fn=fn_name, n=self._specs[0].nodes)

        async def _broadcast() -> list[T]:
            assert self._pool_ref is not None
            result = await self._system.ask(  # type: ignore[union-attr]
                self._pool_ref,
                lambda reply_to: SubmitBroadcast(
                    fn=fn, args=pending.args, kwargs=pending.kwargs,
                    reply_to=reply_to, timeout=timeout,
                ),
                timeout=timeout + 30,
            )
            return [self._unwrap_broadcast_result(v) for v in result]

        return self._run_sync(_broadcast())

    def run_parallel(
        self, group: PendingFunctionGroup
    ) -> tuple[Any, ...] | Generator[Any, None, None]:
        if not self._active or self._pool_ref is None or self._system is None:
            raise RuntimeError("Pool is not active")
        logger.debug("Running {n} tasks in parallel", n=len(group.items))
        if group.stream:
            return self._run_parallel_stream(group)

        async def _run_parallel() -> tuple[Any, ...]:
            assert self._pool_ref is not None
            tasks = [
                self._system.ask(  # type: ignore[union-attr]
                    self._pool_ref, self._submit(p), timeout=self._resolve_timeout(p),
                )
                for p in group.items
            ]
            coro = asyncio.gather(*tasks)
            group_timeout = (
                group.timeout if group.timeout is not None
                else self.default_compute_timeout
            )
            results = await asyncio.wait_for(coro, timeout=group_timeout)
            return tuple(self._unwrap_result(r) for r in results)

        return self._run_sync(_run_parallel())

    def _run_parallel_stream(self, group: PendingFunctionGroup) -> Generator[Any, None, None]:
        q: queue.Queue[Any] = queue.Queue()
        sentinel = object()

        @dataclass(frozen=True, slots=True)
        class _StreamError:
            exception: BaseException

        async def _feed_queue() -> None:
            assert self._pool_ref is not None
            tasks = [
                asyncio.ensure_future(
                    self._system.ask(  # type: ignore[union-attr]
                        self._pool_ref, self._submit(p), timeout=self._resolve_timeout(p),
                    )
                )
                for p in group.items
            ]
            try:
                if group.ordered:
                    for task in tasks:
                        result = await task
                        q.put(self._unwrap_result(result))
                else:
                    for coro in asyncio.as_completed(tasks):
                        result = await coro
                        q.put(self._unwrap_result(result))
            except BaseException as exc:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                q.put(_StreamError(exception=exc))
                return
            q.put(sentinel)

        assert self._loop is not None
        asyncio.run_coroutine_threadsafe(_feed_queue(), self._loop)

        while True:
            item = q.get()
            if item is sentinel:
                break
            if isinstance(item, _StreamError):
                raise item.exception
            yield item

    def map[T, R](
        self,
        fn: Callable[[T], R],
        items: Sequence[T],
    ) -> list[R]:
        if not self._active or self._pool_ref is None or self._system is None:
            raise RuntimeError("Pool is not active")

        async def _map_async() -> list[R]:
            assert self._pool_ref is not None
            tasks = [
                self._system.ask(  # type: ignore[union-attr]
                    self._pool_ref,
                    self._submit(PendingFunction(fn=fn, args=(item,), kwargs={})),
                    timeout=self.default_compute_timeout,
                )
                for item in items
            ]
            return [self._unwrap_result(r) for r in await asyncio.gather(*tasks)]

        return self._run_sync(_map_async())

    def dict(self, name: str, *, consistency: Consistency | None = None) -> DictProxy:
        """Get or create a distributed dict."""
        if self._registry is None:
            raise RuntimeError("Pool is not active")
        logger.debug("Creating distributed dict: {name}", name=name)
        return self._registry.dict(name, consistency=consistency)

    def set(self, name: str, *, consistency: Consistency | None = None) -> SetProxy:
        """Get or create a distributed set."""
        if self._registry is None:
            raise RuntimeError("Pool is not active")
        logger.debug("Creating distributed set: {name}", name=name)
        return self._registry.set(name, consistency=consistency)

    def counter(self, name: str, *, consistency: Consistency | None = None) -> CounterProxy:
        """Get or create a distributed counter."""
        if self._registry is None:
            raise RuntimeError("Pool is not active")
        logger.debug("Creating distributed counter: {name}", name=name)
        return self._registry.counter(name, consistency=consistency)

    def queue(self, name: str) -> QueueProxy:
        """Get or create a distributed queue."""
        if self._registry is None:
            raise RuntimeError("Pool is not active")
        logger.debug("Creating distributed queue: {name}", name=name)
        return self._registry.queue(name)

    def barrier(self, name: str, n: int) -> BarrierProxy:
        """Get or create a distributed barrier."""
        if self._registry is None:
            raise RuntimeError("Pool is not active")
        logger.debug("Creating distributed barrier: {name} (n={n})", name=name, n=n)
        return self._registry.barrier(name, n)

    def lock(self, name: str, timeout: float = 30) -> LockProxy:
        """Get or create a distributed lock."""
        if self._registry is None:
            raise RuntimeError("Pool is not active")
        logger.debug("Creating distributed lock: {name}", name=name)
        return self._registry.lock(name, timeout)

    def current_nodes(self) -> int:
        """Return the number of ready nodes in the pool."""
        if not self._active or self._pool_ref is None or self._system is None:
            raise RuntimeError("Pool is not active")
        result: CurrentNodeCount = self._run_sync(
            self._system.ask(
                self._pool_ref,
                lambda reply_to: GetCurrentNodes(reply_to=reply_to),
                timeout=5.0,
            ),
        )
        return result.ready

    async def _select_offers(
        self,
    ) -> tuple[tuple[Offer, ...], ProviderConfig, Any, PoolSpec]:
        """Rank all offers across specs by price.

        Uses the OfferRepository (SQLite-backed catalog with on-demand live
        fetching) to query and rank offers from all configured providers.

        Returns (offers_sorted, provider_config, cloud_provider, pool_spec).
        """
        from skyward.offers import OfferRepository, to_offer

        specs = self._build_specs()

        provider_instances: dict[str, Any] = {}
        config_by_type: dict[str, ProviderConfig] = {}
        for s in specs:
            ptype = s.provider.type
            if ptype not in provider_instances:
                provider_instances[ptype] = await s.provider.create_provider()
                config_by_type[ptype] = s.provider

        repo = await OfferRepository.create(
            providers=list(provider_instances.values()),
        )

        ranked: list[tuple[float, Offer, ProviderConfig, PoolSpec]] = []

        for s in specs:
            accel = s.accelerator
            ptype = s.provider.type

            query = (
                repo.accelerator(accel.name).provider(ptype) if accel
                else repo.provider(ptype).cpu_only()
            )

            if accel and accel.memory:
                mem = accel.memory.upper().removesuffix("GB")
                if mem.isdigit():
                    query = query.accelerator_memory(int(mem))

            if s.vcpus:
                query = query.vcpus(s.vcpus)
            if s.memory_gb:
                query = query.memory(s.memory_gb)
            if s.max_hourly_cost:
                query = query.max_price(s.max_hourly_cost)

            query = query.allocation(s.allocation)
            use_spot = s.allocation in ("spot", "spot-if-available")

            catalog_offers = await query.cheapest(20)
            if not catalog_offers:
                logger.warning(
                    "No offers from {provider} for accelerator={acc}",
                    provider=ptype, acc=accel.name if accel else "none",
                )
                continue

            provider_config = s.provider
            region = s.region or catalog_offers[0].region

            match s.nodes:
                case (min_n, max_n):
                    spec_nodes = min_n
                    spec_min = min_n
                    spec_max = max_n
                case int(n):
                    spec_nodes = n
                    spec_min = self._scaling[0] if self._scaling else None
                    spec_max = self._scaling[1] if self._scaling else None

            pool_spec = PoolSpec(
                nodes=spec_nodes,
                accelerator=accel,
                region=region,
                vcpus=s.vcpus,
                memory_gb=s.memory_gb,
                architecture=s.architecture,
                allocation=s.allocation,
                image=self.image,
                ttl=s.ttl,
                worker=self.worker,
                provider=provider_config.type,  # type: ignore[arg-type]
                max_hourly_cost=s.max_hourly_cost,
                ssh_timeout=float(self.ssh_timeout),
                ssh_retry_interval=float(self.ssh_retry_interval),
                provision_retry_delay=self.provision_retry_delay,
                max_provision_attempts=self.max_provision_attempts,
                volumes=self.volumes,
                min_nodes=spec_min,
                max_nodes=spec_max,
                autoscale_cooldown=self.autoscale_cooldown,
                autoscale_idle_timeout=self.autoscale_idle_timeout,
                reconcile_tick_interval=self.reconcile_tick_interval,
                plugins=self._plugins,
            )

            for co in catalog_offers:
                offer = to_offer(co)
                price_raw = offer.spot_price if use_spot else offer.on_demand_price
                price = price_raw if price_raw is not None else float("inf")
                ranked.append((price, offer, provider_config, pool_spec))

        repo.close()

        if not ranked:
            raise RuntimeError("No offers found across all specs")

        ranked.sort(key=lambda x: x[0])
        offers = tuple(r[1] for r in ranked)
        _, _, best_config, best_spec = ranked[0]

        cloud_provider = provider_instances[best_config.type]

        logger.info(
            "Selected: {provider} {instance} in {region} (${price}/hr)",
            provider=best_config.type, instance=offers[0].instance_type.name,
            region=best_spec.region, price=offers[0].spot_price or offers[0].on_demand_price or 0,
        )

        return offers, best_config, cloud_provider, best_spec

    async def _start_async(self) -> None:
        """Start pool asynchronously using actors (zero bus)."""
        from skyward.actors.messages import PoolStarted, StartPool
        from skyward.actors.pool import pool_actor

        offers, provider_config, cloud_provider, spec = await self._select_offers()

        best = offers[0]
        logger.info(
            "Ranked {n} offers, best: {offer_id} ({name}, ${price}/hr)",
            n=len(offers), offer_id=best.id, name=best.instance_type.name,
            price=best.spot_price or best.on_demand_price or 0,
        )

        self._spec = spec
        logger.debug(
            "Built PoolSpec: {nodes} nodes, region={region}, allocation={alloc}",
            nodes=spec.nodes, region=spec.region, alloc=spec.allocation,
        )

        self._system = ActorSystem("skyward", config=CastyConfig(
            suppress_dead_letters_on_shutdown=True
        ))

        await self._system.__aenter__()

        if self._app is not None:
            self._app.setup(self._system, spec)

        spy = self._app.spy if self._app is not None else None

        pool_behavior = pool_actor()
        if spy is not None:
            pool_behavior = Behaviors.spy(pool_behavior, spy, spy_children=True)

        logger.debug(
            "Spawning pool actor (spy={spy})",
            spy=spy is not None,
        )
        pool_ref: ActorRef[PoolMsg] = self._system.spawn(pool_behavior, "pool")
        self._pool_ref = pool_ref

        logger.info(
            "Waiting for pool to start (timeout={timeout}s)",
            timeout=self.provision_timeout,
        )
        result: PoolStarted | ProvisionFailed = await self._system.ask(
            pool_ref,
            lambda reply_to: StartPool(
                spec=spec,
                provider_config=provider_config,
                provider=cloud_provider,
                offers=offers,
                reply_to=reply_to,
            ),
            timeout=float(self.provision_timeout),
        )
        match result:
            case ProvisionFailed(reason=reason):
                raise RuntimeError(
                    f"Pool provisioning failed: {reason}"
                )
            case PoolStarted(
                cluster_id=cluster_id, instances=instances, cluster=cluster,
            ):
                logger.info(
                    "Pool started, cluster_id={cid}, instances={n}",
                    cid=cluster_id, n=len(instances),
                )
                self._cluster_id = cluster_id
                self._cluster = cluster
                self.image = self._apply_plugin_transforms(self.image, cluster)
                self._instances = {
                    info.node: info
                    for info in instances
                }

    async def _stop_async(self) -> None:
        """Stop pool asynchronously."""
        if self._pool_ref is not None and self._system is not None:
            from skyward.actors.messages import StopPool
            logger.debug("Sending StopPool to pool actor...")
            await self._system.ask(
                self._pool_ref,
                lambda reply_to: StopPool(reply_to=reply_to),
                timeout=self.shutdown_timeout,
            )
            logger.debug("StopPool ask resolved")

        if self._system is not None:
            logger.debug("Shutting down actor system...")
            await self._system.__aexit__(None, None, None)
            self._system = None
            logger.debug("Actor system stopped")

        _cancel_pending_tasks()

    def _run_loop(self) -> None:
        """Run event loop in background thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()  # type: ignore

    def _run_sync[T](self, coro: Coroutine[Any, Any, T], timeout: float = 3600.0) -> T:
        """Run coroutine synchronously."""
        if self._loop is None:
            raise RuntimeError("Event loop not running")

        future: Future[T] = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def _run_sync_with_timeout[T](self, coro: Coroutine[Any, Any, T], timeout: float) -> T:
        """Run coroutine synchronously with timeout."""
        if self._loop is None:
            raise RuntimeError("Event loop not running")

        future: Future[T] = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def _cleanup(self) -> None:
        """Cleanup resources."""
        logger.debug("Cleaning up event loop and thread")
        loop = self._loop
        thread = self._loop_thread
        if loop is None:
            return

        loop.call_soon_threadsafe(loop.stop)
        logger.debug("Stopping event loop")

        if thread is not None:
            thread.join(timeout=10)
            if thread.is_alive():
                logger.warning("Event loop thread did not stop within 10s")

        if not loop.is_running():
            with suppress(Exception):
                loop.close()

        self._loop = None
        self._loop_thread = None

    @property
    def concurrency(self) -> int:
        """Number of concurrent task slots per node."""
        return self.worker.concurrency

    @property
    def is_active(self) -> bool:
        """True if pool is ready for execution."""
        return self._active

    def __repr__(self) -> str:
        status = "active" if self._active else "inactive"
        first = self._specs[0]
        return f"ComputePool(nodes={first.nodes}, accelerator={first.accelerator}, {status})"

    @classmethod
    def Named(cls, name: str) -> ComputePool:
        from skyward.config import resolve_pool
        return resolve_pool(name)
