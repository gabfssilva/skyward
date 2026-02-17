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
import sys
import threading
import types
from collections.abc import Callable, Coroutine, Generator, Iterator, Sequence
from concurrent.futures import Future
from contextlib import suppress
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Literal, overload

from casty import ActorRef, ActorSystem, Behaviors, CastyConfig

from skyward.accelerators import Accelerator
from skyward.actors.messages import (
    InstanceMetadata,
    PoolMsg,
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

from .spec import DEFAULT_IMAGE, Image, InflightStrategy, PoolSpec

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

        # Override timeout
        result = train(data).with_timeout(600) >> sky
    """

    fn: Callable[..., T]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    timeout: float | None = None

    def with_timeout(self, timeout: float) -> PendingCompute[T]:
        return PendingCompute(fn=self.fn, args=self.args, kwargs=self.kwargs, timeout=timeout)

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
    ordered: bool = True
    timeout: float | None = None

    def with_timeout(self, timeout: float) -> PendingComputeGroup:
        return PendingComputeGroup(
            items=self.items, stream=self.stream,
            ordered=self.ordered, timeout=timeout,
        )

    def __and__(self, other: PendingCompute[Any] | PendingComputeGroup) -> PendingComputeGroup:
        """Add another computation to the group."""
        match other:
            case PendingComputeGroup():
                return PendingComputeGroup(
                    items=(*self.items, *other.items),
                    stream=self.stream, ordered=self.ordered,
                )
            case _:
                return PendingComputeGroup(
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

    def __iter__(self) -> Iterator[PendingCompute[Any]]:
        return iter(self.items)


def gather(
    *pendings: PendingCompute[Any],
    stream: bool = False,
    ordered: bool = True,
) -> PendingComputeGroup:
    """Group computations for parallel execution.

    Example:
        results = gather(task1(), task2(), task3()) >> sky
        # results is a tuple of (result1, result2, result3)

        for result in gather(task1(), task2(), task3(), stream=True) >> sky:
            print(result)  # yields results as they complete

        for result in gather(task1(), task2(), task3(), stream=True, ordered=False) >> sky:
            print(result)  # yields results as they complete, unordered
    """
    return PendingComputeGroup(items=pendings, stream=stream, ordered=ordered)


@overload
def compute[**P, T](fn: Callable[P, T]) -> Callable[P, PendingCompute[T]]: ...

@overload
def compute[**P, T](
    *, timeout: float,
) -> Callable[[Callable[P, T]], Callable[P, PendingCompute[T]]]: ...

def compute[**P, T](
    fn: Callable[P, T] | None = None,
    *,
    timeout: float | None = None,
) -> Callable[P, PendingCompute[T]] | Callable[[Callable[P, T]], Callable[P, PendingCompute[T]]]:
    def decorator(f: Callable[P, T]) -> Callable[P, PendingCompute[T]]:
        @functools.wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> PendingCompute[T]:
            return PendingCompute(fn=f, args=args, kwargs=kwargs, timeout=timeout)
        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator


@dataclass
class ComputePool:
    """
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

    provider: ProviderConfig

    nodes: int = 1
    accelerator: str | Accelerator | None = None
    vcpus: int | None = None
    memory_gb: int | None = None
    architecture: Literal["x86_64", "arm64"] | None = None
    allocation: Literal["spot", "on-demand", "spot-if-available"] = "spot-if-available"

    image: Image = field(default_factory=lambda: DEFAULT_IMAGE)
    ttl: int = 600

    concurrency: int = 1
    max_inflight: int | InflightStrategy | None = None

    logging: LogConfig | bool = True

    max_hourly_cost: float | None = None
    default_compute_timeout: float = 300.0

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
    _console_ref: ActorRef | None = field(default=None, init=False, repr=False)
    _original_stdout: Any = field(default=None, init=False, repr=False)

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
            self._install_stdout_redirect()
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
            self._uninstall_stdout_redirect()

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

    @staticmethod
    def _unwrap_broadcast_result(value: Any) -> Any:
        match value:
            case RuntimeError() as err:
                raise err
            case _:
                return value

    def _unwrap_result(self, result: TaskResult) -> Any:
        return self._unwrap_broadcast_result(result.value)

    def _resolve_timeout(self, pending: PendingCompute[Any]) -> float:
        return pending.timeout if pending.timeout is not None else self.default_compute_timeout

    def _submit(self, pending: PendingCompute[Any]) -> Callable[[ActorRef[Any]], SubmitTask]:
        return lambda reply_to: SubmitTask(
            fn=pending.fn, args=pending.args, kwargs=pending.kwargs,
            reply_to=reply_to,
        )

    def run[T](self, pending: PendingCompute[T]) -> T:
        if not self._active or self._pool_ref is None or self._system is None:
            raise RuntimeError("Pool is not active")

        timeout = self._resolve_timeout(pending)
        logger.debug("Submitting task: {fn}", fn=getattr(pending.fn, "__name__", repr(pending.fn)))
        result: TaskResult = self._run_sync(
            self._system.ask(self._pool_ref, self._submit(pending), timeout=timeout)
        )
        return self._unwrap_result(result)

    def run_async[T](self, pending: PendingCompute[T]) -> Future[T]:
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

    def broadcast[T](self, pending: PendingCompute[T]) -> list[T]:
        if not self._active or self._pool_ref is None or self._system is None:
            raise RuntimeError("Pool is not active")

        timeout = self._resolve_timeout(pending)
        fn_name = getattr(pending.fn, "__name__", repr(pending.fn))
        logger.debug("Broadcasting task: {fn} to {n} nodes", fn=fn_name, n=self.nodes)

        async def _broadcast() -> list[T]:
            assert self._pool_ref is not None
            result = await self._system.ask(  # type: ignore[union-attr]
                self._pool_ref,
                lambda reply_to: SubmitBroadcast(
                    fn=pending.fn, args=pending.args, kwargs=pending.kwargs,
                    reply_to=reply_to,
                ),
                timeout=timeout,
            )
            return list(map(self._unwrap_broadcast_result, result))

        return self._run_sync(_broadcast())

    def run_parallel(
        self, group: PendingComputeGroup
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

    def _run_parallel_stream(self, group: PendingComputeGroup) -> Generator[Any, None, None]:
        q: queue.Queue[Any] = queue.Queue()
        sentinel = object()

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
            if group.ordered:
                for task in tasks:
                    result = await task
                    q.put(self._unwrap_result(result))
            else:
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
                    self._submit(PendingCompute(fn=fn, args=(item,), kwargs={})),
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

    def lock(self, name: str) -> LockProxy:
        """Get or create a distributed lock."""
        if self._registry is None:
            raise RuntimeError("Pool is not active")
        logger.debug("Creating distributed lock: {name}", name=name)
        return self._registry.lock(name)

    def _install_stdout_redirect(self) -> None:
        if self._console_ref is None or self._system is None:
            return
        from skyward.actors.console import LocalOutput

        ref = self._console_ref

        class _ConsoleWriter:
            def __init__(self, original: Any, stream: str = "stdout") -> None:
                self._original = original
                self._stream = stream

            def write(self, s: str) -> int:
                for line in s.splitlines(keepends=True):
                    stripped = line.rstrip()
                    if stripped:
                        ref.tell(LocalOutput(line=stripped, stream=self._stream))
                return len(s)

            def flush(self) -> None:
                pass

            @property
            def encoding(self) -> str:
                return self._original.encoding

            @property
            def errors(self) -> str | None:
                return self._original.errors

            def fileno(self) -> int:
                return self._original.fileno()

            def isatty(self) -> bool:
                return False

        self._original_stdout = sys.stdout
        sys.stdout = _ConsoleWriter(sys.stdout, "stdout")  # type: ignore[assignment]

    def _uninstall_stdout_redirect(self) -> None:
        if self._original_stdout is not None:
            sys.stdout = self._original_stdout
            self._original_stdout = None

    async def _start_async(self) -> None:
        """Start pool asynchronously using actors (zero bus)."""
        from skyward.actors.console import console_actor
        from skyward.actors.messages import PoolStarted, StartPool
        from skyward.actors.pool import pool_actor

        logger.info("Creating cloud provider ({type})", type=self.provider.type)
        cloud_provider = await self.provider.create_provider()
        provider_name = self.provider.type

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
            max_inflight=self.max_inflight,
            provider=provider_name,  # type: ignore[arg-type]
            max_hourly_cost=self.max_hourly_cost,
            ssh_timeout=float(self.ssh_timeout),
            ssh_retry_interval=float(self.ssh_retry_interval),
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

        console_ref = (
            self._system.spawn(console_actor(spec), "console")
            if self.logging
            else None
        )
        self._console_ref = console_ref

        pool_behavior = pool_actor()
        if console_ref is not None:
            pool_behavior = Behaviors.spy(pool_behavior, console_ref, spy_children=True)

        logger.debug(
            "Spawning pool actor (console={console})",
            console=console_ref is not None,
        )
        pool_ref: ActorRef[PoolMsg] = self._system.spawn(pool_behavior, "pool")
        self._pool_ref = pool_ref

        logger.info(
            "Waiting for pool to start (timeout={timeout}s)",
            timeout=self.provision_timeout,
        )
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
        logger.info(
            "Pool started, cluster_id={cid}, instances={n}",
            cid=started.cluster_id, n=len(started.instances),
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
        logger.debug("Cleaning up event loop and thread")
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
            logger.debug("Stopping event loop")
            if self._loop_thread:
                self._loop_thread.join(timeout=5)
                logger.debug("Event loop thread joined")
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
