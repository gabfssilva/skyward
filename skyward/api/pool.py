"""Synchronous facade for v2 async pool.

This module provides the user-facing synchronous API that mirrors v1:

    import skyward as sky

    @sky.compute
    def train(data):
        return model.fit(data)

    @sky.pool(provider=sky.AWS(), accelerator="A100", nodes=4)
    def main():
        result = train(data) >> sky          # execute on one node
        results = train(data) @ sky          # broadcast to all nodes
        a, b = (task1() & task2()) >> sky    # parallel execution

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
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Literal

from casty import ActorRef, ActorSystem, Behaviors, CastyConfig

from skyward.accelerators import Accelerator
from skyward.actors.messages import (
    InstanceMetadata,
    PoolMsg,
    SubmitTask,
    TaskResult,
)
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
from skyward.providers.aws.config import AWS
from skyward.providers.runpod.config import RunPod
from skyward.providers.vastai.config import VastAI
from skyward.providers.verda.config import Verda

from .spec import DEFAULT_IMAGE, Image, PoolSpec

type Provider = AWS | RunPod | VastAI | Verda


_active_pool: ContextVar[ComputePool | None] = ContextVar("active_pool", default=None)


async def _cancel_pending_tasks() -> None:
    current = asyncio.current_task()
    pending = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)


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

    def __rrshift__(self, pending: PendingCompute[Any] | PendingComputeGroup) -> Any:
        """pending >> sky - execute computation(s)."""
        pool = _get_active_pool()
        match pending:
            case PendingComputeGroup():
                return pool.run_parallel(pending)
            case _:
                return pool.run(pending)

    def __rmatmul__(self, pending: PendingCompute[Any]) -> list[Any]:
        """pending @ sky - broadcast to all nodes."""
        pool = _get_active_pool()
        return pool.broadcast(pending)

    def __repr__(self) -> str:
        pool = _active_pool.get()
        if pool:
            return f"<sky: active pool with {pool.nodes} nodes>"
        return "<sky: no active pool>"


sky = _Sky()


@dataclass(frozen=True, slots=True)
class PendingCompute[T]:
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
    """

    fn: Callable[..., T]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

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

    def __and__(self, other: PendingCompute[Any] | PendingComputeGroup) -> PendingComputeGroup:
        """Combine with another computation for parallel execution."""
        match other:
            case PendingComputeGroup():
                return PendingComputeGroup(items=(self, *other.items))
            case _:
                return PendingComputeGroup(items=(self, other))


@dataclass(frozen=True, slots=True)
class PendingComputeGroup:
    """Group of computations for parallel execution.

    Created by using the & operator:
        group = task1() & task2() & task3()
        a, b, c = group >> sky

    Or using gather():
        group = gather(task1(), task2(), task3())
        results = group >> sky
    """

    items: tuple[PendingCompute[Any], ...]
    stream: bool = False

    def __and__(self, other: PendingCompute[Any] | PendingComputeGroup) -> PendingComputeGroup:
        """Add another computation to the group."""
        match other:
            case PendingComputeGroup():
                return PendingComputeGroup(items=(*self.items, *other.items), stream=self.stream)
            case _:
                return PendingComputeGroup(items=(*self.items, other), stream=self.stream)

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

    def __iter__(self) -> Iterator[PendingCompute[Any]]:
        return iter(self.items)


def gather(*pendings: PendingCompute[Any], stream: bool = False) -> PendingComputeGroup:
    """Group computations for parallel execution.

    Example:
        results = gather(task1(), task2(), task3()) >> sky
        # results is a tuple of (result1, result2, result3)

        for result in gather(task1(), task2(), task3(), stream=True) >> sky:
            print(result)  # yields results as they complete
    """
    return PendingComputeGroup(items=pendings, stream=stream)


def compute(fn: Callable[..., Any]) -> Callable[..., PendingCompute[Any]]:
    """Decorator to make a function lazy.

    The decorated function returns a PendingCompute instead of
    executing immediately. Use >> or @ to send it to a pool.

    Example:
        @compute
        def train(model, data):
            return model.fit(data)

        result = train(my_model, my_data) >> sky  # execute on one node
        results = train(my_model, my_data) @ sky  # broadcast to all
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> PendingCompute[Any]:
        return PendingCompute(fn=fn, args=args, kwargs=kwargs)

    return wrapper


def _provider_name(provider: Provider) -> str:
    match provider:
        case AWS():
            return "aws"
        case VastAI():
            return "vastai"
        case Verda():
            return "verda"
        case RunPod():
            return "runpod"
        case _:
            return "unknown"


@dataclass
class ComputePool:
    """Synchronous ComputePool facade.

    Wraps v2's async ComputePool with a synchronous API. Uses a
    dedicated background thread for the asyncio event loop.

    Args:
        provider: Cloud provider configuration (AWS, etc.).
        nodes: Number of nodes to provision.
        accelerator: GPU/accelerator type.
        image: Environment specification.
        region: Cloud region.
        allocation: Instance allocation strategy.
        timeout: Provisioning timeout in seconds.

    Example:
        with pool(provider=AWS(), accelerator="A100") as p:
            result = train(data) >> p

        # Or with decorator:
        @pool(provider=AWS(), accelerator="A100")
        def main():
            return train(data) >> sky
    """

    provider: Provider

    nodes: int = 1
    accelerator: str | Accelerator | None = None
    vcpus: int | None = None
    memory_gb: int | None = None
    architecture: Literal["x86_64", "arm64"] | None = None
    allocation: Literal["spot", "on-demand", "spot-if-available"] = "spot-if-available"

    image: Image = field(default_factory=lambda: DEFAULT_IMAGE)
    ttl: int = 600

    concurrency: int = 1

    panel: bool = True

    logging: LogConfig | bool = True

    max_hourly_cost: float | None = None

    provision_timeout: int = 300
    ssh_timeout: int = 300
    ssh_retry_interval: int = 2

    _log_handler_ids: list[int] = field(default_factory=list, init=False, repr=False)
    _loop: asyncio.AbstractEventLoop | None = field(default=None, init=False, repr=False)
    _loop_thread: threading.Thread | None = field(default=None, init=False, repr=False)

    _active: bool = field(default=False, init=False, repr=False)
    _context_token: Token[ComputePool | None] | None = field(default=None, init=False, repr=False)
    _registry: DistributedRegistry | None = field(default=None, init=False, repr=False)
    _system: ActorSystem | None = field(default=None, init=False, repr=False)
    _pool_ref: ActorRef[PoolMsg] | None = field(default=None, init=False, repr=False)
    _cluster_id: str = field(default="", init=False, repr=False)
    _instances: dict[int, InstanceMetadata] = field(default_factory=dict, init=False, repr=False)
    _spec: PoolSpec | None = field(default=None, init=False, repr=False)

    def __enter__(self) -> ComputePool:
        """Start pool and provision resources."""
        if self.logging:
            match self.logging:
                case True:
                    log_config = LogConfig(console=not self.panel)
                case _:
                    log_config = LogConfig(
                        level=self.logging.level,
                        file=self.logging.file,
                        console=self.logging.console and not self.panel,
                        rotation=self.logging.rotation,
                        retention=self.logging.retention,
                    )
            self._log_handler_ids = setup_logging(log_config)

        logger.info("Starting pool with {n} nodes ({accel})", n=self.nodes, accel=self.accelerator)

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="skyward-event-loop",
        )
        self._loop_thread.start()

        try:
            self._run_sync(self._start_async())
            self._active = True
            self._context_token = _active_pool.set(self)
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
                self._run_sync_with_timeout(self._stop_async(), timeout=30.0)
        except TimeoutError:
            logger.warning("Pool stop timed out after 30s, forcing cleanup")
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
            self._cleanup()
            logger.info("Pool stopped")

            if self._log_handler_ids:
                teardown_logging(self._log_handler_ids)

    def _serialize_pending(self, pending: PendingCompute[Any]) -> bytes:
        from skyward.infra.serialization import serialize
        return serialize({"fn": pending.fn, "args": pending.args, "kwargs": pending.kwargs})

    def _unwrap_result(self, result: TaskResult) -> Any:
        match result.value:
            case RuntimeError() as err:
                raise err
            case value:
                return value

    def run[T](self, pending: PendingCompute[T]) -> T:
        if not self._active or self._pool_ref is None or self._system is None:
            raise RuntimeError("Pool is not active")

        fn_bytes = self._serialize_pending(pending)
        result: TaskResult = self._run_sync(
            self._system.ask(
                self._pool_ref,
                lambda reply_to: SubmitTask(fn_bytes=fn_bytes, reply_to=reply_to),
                timeout=600.0,
            )
        )
        return self._unwrap_result(result)

    def run_async[T](self, pending: PendingCompute[T]) -> Future[T]:
        if not self._active or self._pool_ref is None or self._system is None or self._loop is None:
            raise RuntimeError("Pool is not active")

        fn_bytes = self._serialize_pending(pending)

        async def _run() -> T:
            assert self._pool_ref is not None
            result: TaskResult = await self._system.ask(  # type: ignore[union-attr]
                self._pool_ref,
                lambda reply_to: SubmitTask(fn_bytes=fn_bytes, reply_to=reply_to),
                timeout=600.0,
            )
            return self._unwrap_result(result)

        return asyncio.run_coroutine_threadsafe(_run(), self._loop)

    def broadcast[T](self, pending: PendingCompute[T]) -> list[T]:
        if not self._active or self._pool_ref is None or self._system is None:
            raise RuntimeError("Pool is not active")

        fn_bytes = self._serialize_pending(pending)

        async def _broadcast() -> list[T]:
            assert self._pool_ref is not None
            tasks = [
                self._system.ask(  # type: ignore[union-attr]
                    self._pool_ref,
                    lambda reply_to: SubmitTask(fn_bytes=fn_bytes, reply_to=reply_to),
                    timeout=600.0,
                )
                for _ in range(self.nodes)
            ]
            results = await asyncio.gather(*tasks)
            return [self._unwrap_result(r) for r in results]

        return self._run_sync(_broadcast())

    def run_parallel(
        self, group: PendingComputeGroup
    ) -> tuple[Any, ...] | Generator[Any, None, None]:
        if not self._active or self._pool_ref is None or self._system is None:
            raise RuntimeError("Pool is not active")

        if group.stream:
            return self._run_parallel_stream(group)

        async def _run_parallel() -> tuple[Any, ...]:
            assert self._pool_ref is not None
            tasks = [
                self._system.ask(  # type: ignore[union-attr]
                    self._pool_ref,
                    lambda reply_to, p=p: SubmitTask(
                        fn_bytes=self._serialize_pending(p),
                        reply_to=reply_to,
                    ),
                    timeout=600.0,
                )
                for p in group.items
            ]
            results = await asyncio.gather(*tasks)
            return tuple(self._unwrap_result(r) for r in results)

        return self._run_sync(_run_parallel())

    def _run_parallel_stream(self, group: PendingComputeGroup) -> Generator[Any, None, None]:
        q: queue.Queue[Any] = queue.Queue()
        sentinel = object()

        async def _feed_queue() -> None:
            assert self._pool_ref is not None
            tasks = [
                asyncio.ensure_future(
                    self._system.ask(  # type: ignore[union-attr]
                        self._pool_ref,
                        lambda reply_to, p=p: SubmitTask(
                            fn_bytes=self._serialize_pending(p),
                            reply_to=reply_to,
                        ),
                        timeout=600.0,
                    )
                )
                for p in group.items
            ]
            for coro in asyncio.as_completed(tasks):
                result = await coro
                q.put(self._unwrap_result(result))
            q.put(sentinel)

        assert self._loop is not None
        asyncio.run_coroutine_threadsafe(_feed_queue(), self._loop)

        while True:
            item = q.get()
            if item is sentinel:
                break
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
                    lambda reply_to, item=item: SubmitTask(
                        fn_bytes=self._serialize_pending(
                            PendingCompute(fn=fn, args=(item,), kwargs={})
                        ),
                        reply_to=reply_to,
                    ),
                    timeout=600.0,
                )
                for item in items
            ]
            return [self._unwrap_result(r) for r in await asyncio.gather(*tasks)]

        return self._run_sync(_map_async())

    def dict(self, name: str, *, consistency: Consistency | None = None) -> DictProxy:
        """Get or create a distributed dict."""
        if self._registry is None:
            raise RuntimeError("Pool is not active")
        return self._registry.dict(name, consistency=consistency)

    def set(self, name: str, *, consistency: Consistency | None = None) -> SetProxy:
        """Get or create a distributed set."""
        if self._registry is None:
            raise RuntimeError("Pool is not active")
        return self._registry.set(name, consistency=consistency)

    def counter(self, name: str, *, consistency: Consistency | None = None) -> CounterProxy:
        """Get or create a distributed counter."""
        if self._registry is None:
            raise RuntimeError("Pool is not active")
        return self._registry.counter(name, consistency=consistency)

    def queue(self, name: str) -> QueueProxy:
        """Get or create a distributed queue."""
        if self._registry is None:
            raise RuntimeError("Pool is not active")
        return self._registry.queue(name)

    def barrier(self, name: str, n: int) -> BarrierProxy:
        """Get or create a distributed barrier."""
        if self._registry is None:
            raise RuntimeError("Pool is not active")
        return self._registry.barrier(name, n)

    def lock(self, name: str) -> LockProxy:
        """Get or create a distributed lock."""
        if self._registry is None:
            raise RuntimeError("Pool is not active")
        return self._registry.lock(name)

    async def _start_async(self) -> None:
        """Start pool asynchronously using actors (zero bus)."""
        from skyward.actors.messages import PoolStarted, StartPool
        from skyward.actors.panel import panel_actor
        from skyward.actors.pool import pool_actor

        cloud_provider = self.provider.create_provider()
        provider_name = _provider_name(self.provider)

        region = getattr(self.provider, "region", "unknown")

        spec = PoolSpec(
            nodes=self.nodes,
            accelerator=self.accelerator,
            region=region,
            vcpus=self.vcpus,
            memory_gb=self.memory_gb,
            architecture=self.architecture,
            allocation=self.allocation,
            image=self.image,
            ttl=self.ttl,
            concurrency=self.concurrency,
            provider=provider_name,  # type: ignore[arg-type]
            max_hourly_cost=self.max_hourly_cost,
            ssh_timeout=float(self.ssh_timeout),
            ssh_retry_interval=float(self.ssh_retry_interval),
        )
        self._spec = spec

        self._system = ActorSystem("skyward", config=CastyConfig(
            suppress_dead_letters_on_shutdown=True
        ))

        await self._system.__aenter__()

        panel_ref = (
            self._system.spawn(panel_actor(spec), "panel")
            if self.panel
            else None
        )

        pool_behavior = pool_actor()
        spy = panel_ref is not None
        pool_ref: ActorRef[PoolMsg] = self._system.spawn(
            Behaviors.spy(pool_behavior, panel_ref, spy_children=True)
            if spy
            else pool_behavior,
            "pool",
        )
        self._pool_ref = pool_ref

        started: PoolStarted = await self._system.ask(
            pool_ref,
            lambda reply_to: StartPool(
                spec=spec,
                provider_config=self.provider,
                provider=cloud_provider,
                reply_to=reply_to,
            ),
            timeout=float(self.provision_timeout),
        )
        self._cluster_id = started.cluster_id
        self._instances = {
            info.node: info
            for info in started.instances
        }


    async def _stop_async(self) -> None:
        """Stop pool asynchronously."""
        if self._pool_ref is not None and self._system is not None:
            from skyward.actors.messages import StopPool
            logger.debug("Sending StopPool to pool actor...")
            await self._system.ask(
                self._pool_ref,
                lambda reply_to: StopPool(reply_to=reply_to),
                timeout=30.0,
            )
            logger.debug("StopPool completed, cluster destroyed")

        if self._system is not None:
            await self._system.__aexit__(None, None, None)
            self._system = None

        await _cancel_pending_tasks()

    def _run_loop(self) -> None:
        """Run event loop in background thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()  # type: ignore

    def _run_sync[T](self, coro: Coroutine[Any, Any, T]) -> T:
        """Run coroutine synchronously."""
        if self._loop is None:
            raise RuntimeError("Event loop not running")

        future: Future[T] = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def _run_sync_with_timeout[T](self, coro: Coroutine[Any, Any, T], timeout: float) -> T:
        """Run coroutine synchronously with timeout."""
        if self._loop is None:
            raise RuntimeError("Event loop not running")

        future: Future[T] = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def _cleanup(self) -> None:
        """Cleanup resources."""
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread:
                self._loop_thread.join(timeout=5)
            with suppress(Exception):
                self._loop.close()
            self._loop = None
            self._loop_thread = None

    @property
    def is_active(self) -> bool:
        """True if pool is ready for execution."""
        return self._active

    def __repr__(self) -> str:
        status = "active" if self._active else "inactive"
        return f"ComputePool(nodes={self.nodes}, accelerator={self.accelerator}, {status})"

    @classmethod
    def Named(cls, name: str) -> ComputePool:
        from skyward.config import resolve_pool
        return resolve_pool(name)


class _PoolFactory:
    """Factory that can be used as context manager or decorator."""

    def __init__(
        self,
        provider: Provider | None = None,
        nodes: int = 1,
        accelerator: str | Accelerator | None = None,
        vcpus: int | None = None,
        memory_gb: int | None = None,
        architecture: Literal["x86_64", "arm64"] | None = None,
        image: Image | None = None,
        allocation: Literal["spot", "on-demand", "spot-if-available"] = "spot-if-available",
        ttl: int = 600,
        panel: bool = True,
        logging: LogConfig | bool = True,
        max_hourly_cost: float | None = None,
        provision_timeout: int = 300,
        ssh_timeout: int = 300,
        ssh_retry_interval: int = 2,
    ) -> None:
        self._provider = provider
        self._nodes = nodes
        self._accelerator = accelerator
        self._vcpus = vcpus
        self._memory_gb = memory_gb
        self._architecture = architecture
        self._image = image
        self._allocation = allocation
        self._ttl = ttl
        self._panel = panel
        self._logging = logging
        self._max_hourly_cost = max_hourly_cost
        self._provision_timeout = provision_timeout
        self._ssh_timeout = ssh_timeout
        self._ssh_retry_interval = ssh_retry_interval
        self._pool: ComputePool | None = None

    def _create_pool(self) -> ComputePool:
        """Create the underlying pool."""
        provider = self._provider or AWS()
        return ComputePool(
            provider=provider,
            nodes=self._nodes,
            accelerator=self._accelerator,
            vcpus=self._vcpus,
            memory_gb=self._memory_gb,
            architecture=self._architecture,  # type: ignore[arg-type]
            allocation=self._allocation,  # type: ignore[arg-type]
            image=self._image or DEFAULT_IMAGE,
            ttl=self._ttl,
            panel=self._panel,
            logging=self._logging,
            max_hourly_cost=self._max_hourly_cost,
            provision_timeout=self._provision_timeout,
            ssh_timeout=self._ssh_timeout,
            ssh_retry_interval=self._ssh_retry_interval,
        )

    def __enter__(self) -> ComputePool:
        """Use as context manager."""
        self._pool = self._create_pool()
        return self._pool.__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager."""
        if self._pool:
            self._pool.__exit__(exc_type, exc_val, exc_tb)
            self._pool = None



def pool(
    provider: Provider | None = None,
    nodes: int = 1,
    accelerator: str | Accelerator | None = None,
    vcpus: int | None = None,
    memory_gb: int | None = None,
    architecture: Literal["x86_64", "arm64"] | None = None,
    image: Image | None = None,
    allocation: Literal["spot", "on-demand", "spot-if-available"] = "spot-if-available",
    ttl: int = 600,
    panel: bool = True,
    logging: LogConfig | bool = True,
    max_hourly_cost: float | None = None,
    provision_timeout: int = 300,
    ssh_timeout: int = 300,
    ssh_retry_interval: int = 2,
) -> _PoolFactory:
    """Create a compute pool context manager.

    Example:
        with pool(provider=AWS(), accelerator="A100") as p:
            result = train(data) >> p
    """
    return _PoolFactory(
        provider=provider,
        nodes=nodes,
        accelerator=accelerator,
        vcpus=vcpus,
        memory_gb=memory_gb,
        architecture=architecture,
        image=image,
        allocation=allocation,
        ttl=ttl,
        panel=panel,
        logging=logging,
        max_hourly_cost=max_hourly_cost,
        provision_timeout=provision_timeout,
        ssh_timeout=ssh_timeout,
        ssh_retry_interval=ssh_retry_interval,
    )
