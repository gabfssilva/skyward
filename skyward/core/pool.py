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
import queue
from collections.abc import Callable, Generator, Sequence
from concurrent.futures import Future
from contextvars import Token
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, overload

from casty import ActorRef, ActorSystem

from skyward.accelerators import Accelerator
from skyward.actors.messages import (
    CurrentNodeCount,
    GetCurrentNodes,
    NodeInstance,
    SubmitBroadcast,
    SubmitTask,
    TaskResult,
)
from skyward.core.provider import ProviderConfig
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

from .context import _active_pool
from .function import PendingFunction, PendingFunctionGroup
from .loop import check_fd_budget, run_sync
from .offers import PoolConfig
from .spec import DEFAULT_IMAGE, Image, PoolSpec, SelectionStrategy, Spec, Volume, Worker

if TYPE_CHECKING:
    from skyward.actors.pool.messages import PoolMsg
    from skyward.core.model import Cluster


class ComputePool:
    """Provision cloud compute and dispatch functions to remote nodes.

    Use as a context manager: provisions on enter, destroys on exit.
    Internally runs an asyncio event loop in a background daemon thread;
    the public API is entirely synchronous.

    Two modes:

    - **Single provider** --- pass ``provider=``, ``accelerator=``, ``nodes=``.
    - **Multi-spec fallback** --- pass ``Spec(...)`` positional args with
      ``selection=``.

    Examples
    --------
    Single provider:

    >>> with ComputePool(provider=AWS(), accelerator=A100(), nodes=4) as pool:
    ...     result = train(data) >> pool       # one node (round-robin)
    ...     results = train(data) @ pool       # broadcast to all nodes
    ...     a, b = (task1() & task2()) >> pool  # parallel execution

    Multi-spec with fallback (cheapest across providers):

    >>> with ComputePool(
    ...     Spec(provider=VastAI(), accelerator=A100()),
    ...     Spec(provider=AWS(), accelerator=A100()),
    ...     selection="cheapest",
    ... ) as pool:
    ...     result = train(data) >> pool
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
        self._session: Any = None
        self._owns_session: bool = False

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            raise RuntimeError("Event loop not running")
        return self._loop

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

        from skyward.core.context import get_session
        from skyward.core.session import Session

        session = get_session()
        if session is None:
            session = Session(
                console=True,
                logging=False,
                shutdown_timeout=self.shutdown_timeout,
            )
            session.__enter__()
            self._owns_session = True
        self._session = session
        self._loop = session._loop
        self._system = session._system

        check_fd_budget(fd_nodes)

        try:
            self._spawn_via_session(session)
            self._active = True
            self._context_token = _active_pool.set(self)

            assert self._cluster is not None
            for plugin in self._plugins:
                if plugin.around_client is not None:
                    ctx = plugin.around_client(self, self._cluster)
                    ctx.__enter__()
                    self._plugin_client_contexts.append(ctx)

            logger.info("Pool ready")
        except Exception as e:
            logger.exception("Error starting pool: {err}", err=e)
            if self._owns_session:
                session.__exit__(None, None, None)
                self._session = None
                self._owns_session = False
            self._loop = None
            self._system = None
            if self._log_handler_ids:
                teardown_logging(self._log_handler_ids)
            raise

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Stop pool and release resources."""
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

            if self._active and self._loop is not None:
                run_sync(self._loop, self._stop_pool_actor(), timeout=self.shutdown_timeout)
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

            if self._owns_session and self._session is not None:
                self._session.__exit__(None, None, None)
                self._session = None
                self._owns_session = False

            self._loop = None
            self._system = None
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

    def _assert_active(self) -> None:
        if not self._active or self._pool_ref is None or self._system is None:
            raise RuntimeError("Pool is not active")

    def _get_system_and_ref(self) -> tuple[ActorSystem, ActorRef[PoolMsg]]:
        self._assert_active()
        system = self._system
        ref = self._pool_ref
        if system is None or ref is None:
            raise RuntimeError("Pool is not active")
        return system, ref

    def run[T](self, pending: PendingFunction[T]) -> T:
        """Execute a pending function on one node via round-robin scheduling.

        This is the method behind the ``task() >> pool`` operator.

        Parameters
        ----------
        pending
            Frozen snapshot of the function, args, and kwargs to execute.

        Returns
        -------
        T
            The return value of the remote function call.

        Raises
        ------
        RuntimeError
            When the pool is not active or the remote function raises.

        Examples
        --------
        >>> result = train(data) >> pool
        """
        system, ref = self._get_system_and_ref()

        timeout = self._resolve_timeout(pending)
        logger.debug("Submitting task: {fn}", fn=getattr(pending.fn, "__name__", repr(pending.fn)))
        result: TaskResult = run_sync(
            self._get_loop(),
            system.ask(ref, self._submit(pending), timeout=timeout),
        )
        return self._unwrap_result(result)

    def run_async[T](self, pending: PendingFunction[T]) -> Future[T]:
        """Submit a pending function for asynchronous execution, returning a future.

        This is the method behind the ``task() > pool`` operator. The function
        is dispatched to one node via round-robin but returns immediately
        without blocking.

        Parameters
        ----------
        pending
            Frozen snapshot of the function, args, and kwargs to execute.

        Returns
        -------
        Future[T]
            A concurrent future that resolves to the remote return value.

        Raises
        ------
        RuntimeError
            When the pool is not active.

        Examples
        --------
        >>> future = train(data) > pool
        >>> result = future.result()
        """
        self._assert_active()

        timeout = self._resolve_timeout(pending)
        fn_name = getattr(pending.fn, "__name__", repr(pending.fn))
        logger.debug("Submitting async task: {fn}", fn=fn_name)
        loop = self._get_loop()

        async def _run() -> T:
            assert self._pool_ref is not None
            result: TaskResult = await self._system.ask(  # type: ignore[union-attr]
                self._pool_ref, self._submit(pending), timeout=timeout,
            )
            return self._unwrap_result(result)

        return asyncio.run_coroutine_threadsafe(_run(), loop)

    def broadcast[T](self, pending: PendingFunction[T]) -> list[T]:
        """Execute a pending function on ALL nodes simultaneously.

        This is the method behind the ``task() @ pool`` operator. The function
        is sent to every ready node and all results are collected.

        Parameters
        ----------
        pending
            Frozen snapshot of the function, args, and kwargs to execute.

        Returns
        -------
        list[T]
            One result per node, in node order.

        Raises
        ------
        RuntimeError
            When the pool is not active or any node raises.

        Examples
        --------
        >>> results = train(data) @ pool
        """
        self._assert_active()

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

        return run_sync(self._get_loop(), _broadcast())

    def run_parallel(
        self, group: PendingFunctionGroup
    ) -> tuple[Any, ...] | Generator[Any, None, None]:
        """Execute a group of pending functions concurrently across nodes.

        This is the method behind ``(task1() & task2()) >> pool``. Each
        function in the group is dispatched as a separate task. When
        ``group.stream`` is set, return a generator that yields results
        as they complete instead of waiting for all tasks.

        Parameters
        ----------
        group
            A group of pending functions, created via the ``&`` operator
            or ``sky.gather()``.

        Returns
        -------
        tuple[Any, ...] | Generator[Any, None, None]
            A tuple of results (one per task) when not streaming, or a
            generator yielding results as they arrive when streaming.

        Raises
        ------
        RuntimeError
            When the pool is not active.

        Examples
        --------
        >>> a, b = (task1() & task2()) >> pool
        >>> for result in sky.gather(t1(), t2(), stream=True) >> pool:
        ...     print(result)
        """
        self._assert_active()
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

        return run_sync(self._get_loop(), _run_parallel())

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
        """Apply a function to each item, distributing across nodes.

        Each item becomes a separate task dispatched round-robin. Results
        are returned in the same order as the input items.

        Parameters
        ----------
        fn
            Function to apply to each item.
        items
            Sequence of inputs to map over.

        Returns
        -------
        list[R]
            Ordered list of results, one per input item.

        Raises
        ------
        RuntimeError
            When the pool is not active.

        Examples
        --------
        >>> results = pool.map(process, [chunk1, chunk2, chunk3])
        """
        self._assert_active()

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

        return run_sync(self._get_loop(), _map_async())

    def dict(self, name: str, *, consistency: Consistency | None = None) -> DictProxy:
        """Get or create a distributed dictionary shared across all nodes.

        Parameters
        ----------
        name
            Unique identifier for this collection.
        consistency
            Consistency level. ``None`` uses the system default.

        Returns
        -------
        DictProxy
            Synchronous dict-like proxy backed by the actor system.

        Raises
        ------
        RuntimeError
            When the pool is not active.

        Examples
        --------
        >>> metrics = pool.dict("metrics")
        >>> metrics["loss"] = 0.5
        """
        if self._registry is None:
            raise RuntimeError("Pool is not active")
        logger.debug("Creating distributed dict: {name}", name=name)
        return self._registry.dict(name, consistency=consistency)

    def set(self, name: str, *, consistency: Consistency | None = None) -> SetProxy:
        """Get or create a distributed set shared across all nodes.

        Parameters
        ----------
        name
            Unique identifier for this collection.
        consistency
            Consistency level. ``None`` uses the system default.

        Returns
        -------
        SetProxy
            Synchronous set-like proxy backed by the actor system.

        Raises
        ------
        RuntimeError
            When the pool is not active.

        Examples
        --------
        >>> visited = pool.set("visited")
        >>> visited.add("node-1")
        """
        if self._registry is None:
            raise RuntimeError("Pool is not active")
        logger.debug("Creating distributed set: {name}", name=name)
        return self._registry.set(name, consistency=consistency)

    def counter(self, name: str, *, consistency: Consistency | None = None) -> CounterProxy:
        """Get or create a distributed counter shared across all nodes.

        Parameters
        ----------
        name
            Unique identifier for this collection.
        consistency
            Consistency level. ``None`` uses the system default.

        Returns
        -------
        CounterProxy
            Synchronous counter proxy backed by the actor system.

        Raises
        ------
        RuntimeError
            When the pool is not active.

        Examples
        --------
        >>> steps = pool.counter("steps")
        >>> steps.increment()
        >>> steps.value()
        1
        """
        if self._registry is None:
            raise RuntimeError("Pool is not active")
        logger.debug("Creating distributed counter: {name}", name=name)
        return self._registry.counter(name, consistency=consistency)

    def queue(self, name: str) -> QueueProxy:
        """Get or create a distributed queue shared across all nodes.

        Parameters
        ----------
        name
            Unique identifier for this collection.

        Returns
        -------
        QueueProxy
            Synchronous queue proxy backed by the actor system.

        Raises
        ------
        RuntimeError
            When the pool is not active.

        Examples
        --------
        >>> work = pool.queue("work")
        >>> work.put("item-1")
        >>> work.get()
        'item-1'
        """
        if self._registry is None:
            raise RuntimeError("Pool is not active")
        logger.debug("Creating distributed queue: {name}", name=name)
        return self._registry.queue(name)

    def barrier(self, name: str, n: int) -> BarrierProxy:
        """Get or create a distributed barrier shared across all nodes.

        All participants must call ``wait()`` before any can proceed.

        Parameters
        ----------
        name
            Unique identifier for this barrier.
        n
            Number of participants that must arrive before the barrier opens.

        Returns
        -------
        BarrierProxy
            Synchronous barrier proxy backed by the actor system.

        Raises
        ------
        RuntimeError
            When the pool is not active.

        Examples
        --------
        >>> sync = pool.barrier("epoch-sync", n=pool.current_nodes())
        >>> sync.wait()
        """
        if self._registry is None:
            raise RuntimeError("Pool is not active")
        logger.debug("Creating distributed barrier: {name} (n={n})", name=name, n=n)
        return self._registry.barrier(name, n)

    def lock(self, name: str, timeout: float = 30) -> LockProxy:
        """Get or create a distributed lock shared across all nodes.

        Parameters
        ----------
        name
            Unique identifier for this lock.
        timeout
            Maximum seconds to wait when acquiring. Default ``30``.

        Returns
        -------
        LockProxy
            Synchronous lock proxy backed by the actor system. Supports
            use as a context manager.

        Raises
        ------
        RuntimeError
            When the pool is not active.

        Examples
        --------
        >>> with pool.lock("checkpoint"):
        ...     save_checkpoint(model)
        """
        if self._registry is None:
            raise RuntimeError("Pool is not active")
        logger.debug("Creating distributed lock: {name}", name=name)
        return self._registry.lock(name, timeout)

    def current_nodes(self) -> int:
        """Return the number of ready nodes in the pool.

        Query the pool actor for the current count of nodes that have
        completed bootstrapping and are accepting tasks.

        Returns
        -------
        int
            Number of nodes currently in the ready state.

        Raises
        ------
        RuntimeError
            When the pool is not active.

        Examples
        --------
        >>> pool.current_nodes()
        4
        """
        system, ref = self._get_system_and_ref()
        result: CurrentNodeCount = run_sync(
            self._get_loop(),
            system.ask(
                ref,
                lambda reply_to: GetCurrentNodes(reply_to=reply_to),
                timeout=5.0,
            ),
        )
        return result.ready

    def _pool_config(self) -> PoolConfig:
        return PoolConfig(
            image=self.image,
            worker=self.worker,
            scaling=self._scaling,
            ssh_timeout=self.ssh_timeout,
            ssh_retry_interval=self.ssh_retry_interval,
            provision_retry_delay=self.provision_retry_delay,
            max_provision_attempts=self.max_provision_attempts,
            volumes=self.volumes,
            autoscale_cooldown=self.autoscale_cooldown,
            autoscale_idle_timeout=self.autoscale_idle_timeout,
            reconcile_tick_interval=self.reconcile_tick_interval,
            plugins=self._plugins,
        )

    def _spawn_via_session(self, session: Any) -> None:
        """Select offers and ask the session actor to spawn a pool."""
        pool_ref, spec, cluster_id, cluster, instances = session._spawn_pool(
            self._build_specs(),
            self._pool_config(),
            f"pool-{id(self)}",
            float(self.provision_timeout),
        )

        self._spec = spec
        self._pool_ref = pool_ref
        self._cluster_id = cluster_id
        self._cluster = cluster
        self.image = self._apply_plugin_transforms(self.image, cluster)
        self._instances = {info.node: info for info in instances}

    async def _stop_pool_actor(self) -> None:
        """Stop the pool actor without tearing down the actor system."""
        if self._pool_ref is not None and self._system is not None:
            from skyward.actors.pool.messages import StopPool

            logger.debug("Sending StopPool to pool actor...")
            await self._system.ask(
                self._pool_ref,
                lambda reply_to: StopPool(reply_to=reply_to),
                timeout=self.shutdown_timeout,
            )
            logger.debug("StopPool ask resolved")

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
    def _from_session(
        cls,
        session: Any,
        pool_ref: ActorRef[PoolMsg],
        spec: PoolSpec,
        specs: tuple[Spec, ...],
        plugins: tuple[Plugin, ...],
        cluster_id: str,
        cluster: Cluster[Any],
        instances: tuple[Any, ...],
        image: Image,
        worker: Worker,
        default_compute_timeout: float,
    ) -> ComputePool:
        """Create a ComputePool bound to an existing Session (internal)."""
        pool = cls.__new__(cls)
        pool._specs = specs
        pool._scaling = None
        pool.selection = "cheapest"
        pool.image = image
        pool.worker = worker
        pool.logging = False
        pool.default_compute_timeout = default_compute_timeout
        pool.provision_timeout = 300
        pool.ssh_timeout = 300
        pool.ssh_retry_interval = 2
        pool.provision_retry_delay = 5.0
        pool.max_provision_attempts = 3
        pool.volumes = ()
        pool._plugins = plugins
        pool.autoscale_cooldown = 30.0
        pool.autoscale_idle_timeout = 60.0
        pool.reconcile_tick_interval = 15.0
        pool.shutdown_timeout = 120.0
        pool._log_handler_ids = []
        pool._loop = session._loop
        pool._active = True
        pool._context_token = None
        pool._registry = None
        pool._system = session._system
        pool._pool_ref = pool_ref
        pool._cluster_id = cluster_id
        pool._cluster = cluster
        pool._instances = {info.node: info for info in instances} if instances else {}
        pool._spec = spec
        pool._plugin_client_contexts = []
        pool._session = session
        pool._owns_session = False
        return pool

    @classmethod
    def Named(cls, name: str) -> ComputePool:
        """Create a pool from a named configuration in ``skyward.toml``.

        Look up the pool definition by name in the project's ``skyward.toml``
        or the user-level ``~/.skyward/defaults.toml`` and construct a
        ``ComputePool`` with the resolved settings.

        Parameters
        ----------
        name
            Pool name as defined in the ``[pools.<name>]`` section of
            ``skyward.toml``.

        Returns
        -------
        ComputePool
            A fully configured pool instance (not yet entered).

        Examples
        --------
        >>> with ComputePool.Named("training") as pool:
        ...     result = train(data) >> pool
        """
        from skyward.config import resolve_pool
        return resolve_pool(name)
