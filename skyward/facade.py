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
import threading
import types
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Literal, overload

from injector import Injector
from loguru import logger

from .accelerators import Accelerator
from .bus import AsyncEventBus
from .image import DEFAULT_IMAGE, Image
from .monitors import MonitorModule
from .pool import ComputePool as AsyncPool
from .spec import PoolSpec
from .observability.logging import LogConfig, _setup_logging, _teardown_logging

# Import only config classes - these have NO SDK dependencies
# Handlers and modules are imported lazily in _start_async()
from .providers.aws.config import AWS
from .providers.vastai.config import VastAI
from .providers.verda.config import Verda

# Distributed collections
from .distributed import (
    DistributedRegistry,
    _set_active_registry,
    DictProxy,
    ListProxy,
    SetProxy,
    CounterProxy,
    QueueProxy,
    BarrierProxy,
    LockProxy,
)
from .distributed.types import Consistency

# Type alias for all supported providers
type Provider = AWS | VastAI | Verda


# =============================================================================
# Context Variable for active pool
# =============================================================================

_active_pool: ContextVar[SyncComputePool | None] = ContextVar("active_pool", default=None)


def _get_active_pool() -> SyncComputePool:
    """Get the active pool from context."""
    pool = _active_pool.get()
    if pool is None:
        raise RuntimeError(
            "No active pool. Use within a @pool decorated function or 'with pool(...):' block."
        )
    return pool


# =============================================================================
# Sky Singleton (captures >> and @ operators)
# =============================================================================


class _Sky:
    """Singleton that captures >> and @ operators.

    This allows the v1-style API:
        result = compute_fn(args) >> sky   # execute on one node
        results = compute_fn(args) @ sky   # broadcast to all nodes
    """

    def __rrshift__(self, pending: PendingCompute[Any] | PendingComputeGroup) -> Any:
        """pending >> sky - execute computation(s)."""
        pool = _get_active_pool()
        if isinstance(pending, PendingComputeGroup):
            return pool.run_parallel(pending)
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


# The singleton instance
sky = _Sky()


# =============================================================================
# Pending Compute (lazy evaluation)
# =============================================================================


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

    def __rshift__(self, target: SyncComputePool | _Sky | types.ModuleType) -> T:
        """Execute on pool using >> operator."""
        # Handle case where target is module (import skyward as sky)
        if isinstance(target, types.ModuleType) and hasattr(target, "sky"):
            return target.sky.__rrshift__(self)  # type: ignore
        if isinstance(target, _Sky):
            return target.__rrshift__(self)  # type: ignore
        return target.run(self)  # type: ignore[union-attr]

    def __matmul__(self, target: SyncComputePool | _Sky | types.ModuleType) -> list[T] | tuple[T, ...]:
        """Broadcast to all nodes using @ operator."""
        # Handle case where target is module (import skyward as sky)
        if isinstance(target, types.ModuleType) and hasattr(target, "sky"):
            return target.sky.__rmatmul__(self)
        if isinstance(target, _Sky):
            return target.__rmatmul__(self)
        return target.broadcast(self)  # type: ignore[union-attr]

    def __and__(self, other: PendingCompute[Any] | PendingComputeGroup) -> PendingComputeGroup:
        """Combine with another computation for parallel execution."""
        if isinstance(other, PendingComputeGroup):
            return PendingComputeGroup(items=(self, *other.items))
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

    def __and__(self, other: PendingCompute[Any] | PendingComputeGroup) -> PendingComputeGroup:
        """Add another computation to the group."""
        if isinstance(other, PendingComputeGroup):
            return PendingComputeGroup(items=(*self.items, *other.items))
        return PendingComputeGroup(items=(*self.items, other))

    def __rshift__(self, target: SyncComputePool | _Sky | types.ModuleType) -> tuple[Any, ...]:
        """Execute all computations in parallel using >> operator."""
        # Handle case where target is module (import skyward as sky)
        if isinstance(target, types.ModuleType) and hasattr(target, "sky"):
            return target.sky.__rrshift__(self)  # type: ignore
        if isinstance(target, _Sky):
            return target.__rrshift__(self)  # type: ignore
        return target.run_parallel(self)  # type: ignore[union-attr]

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)


def gather(*pendings: PendingCompute[Any]) -> PendingComputeGroup:
    """Group computations for parallel execution.

    Example:
        results = gather(task1(), task2(), task3()) >> sky
        # results is a tuple of (result1, result2, result3)
    """
    return PendingComputeGroup(items=pendings)


def compute[F: Callable[..., Any]](fn: F) -> Callable[..., PendingCompute[Any]]:
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


# =============================================================================
# Sync Pool Facade
# =============================================================================


@dataclass
class SyncComputePool:
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

    # Required
    provider: Provider

    # Resources
    nodes: int = 1
    accelerator: str | Accelerator | None = None
    vcpus: int | None = None
    memory_gb: int | None = None
    architecture: Literal["x86_64", "arm64"] | None = None
    allocation: Literal["spot", "on-demand", "spot-if-available"] = "spot-if-available"

    # Environment
    image: Image = field(default_factory=lambda: DEFAULT_IMAGE)
    timeout: int = 3600

    # Panel UI
    panel: bool = True  # Enable Rich terminal dashboard

    # Logging
    logging: LogConfig | bool = True  # Logs to .skyward/skyward.log

    # Budget
    max_hourly_cost: float | None = None  # USD/hr for entire cluster

    # Internal
    _log_handler_ids: list[int] = field(default_factory=list, init=False, repr=False)
    _loop: asyncio.AbstractEventLoop | None = field(default=None, init=False, repr=False)
    _loop_thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _async_pool: AsyncPool | None = field(default=None, init=False, repr=False)
    _bus: AsyncEventBus | None = field(default=None, init=False, repr=False)
    _injector: Injector | None = field(default=None, init=False, repr=False)
    _active: bool = field(default=False, init=False, repr=False)
    _context_token: Any = field(default=None, init=False, repr=False)
    _registry: DistributedRegistry | None = field(default=None, init=False, repr=False)

    def __enter__(self) -> SyncComputePool:
        """Start pool and provision resources."""
        # Setup logging BEFORE any logs are emitted
        if self.logging:
            if self.logging is True:
                # Default config - disable console when panel is active
                log_config = LogConfig(console=not self.panel)
            else:
                # User provided config - override console if panel is active
                log_config = LogConfig(
                    level=self.logging.level,
                    file=self.logging.file,
                    console=self.logging.console and not self.panel,
                    rotation=self.logging.rotation,
                    retention=self.logging.retention,
                )
            self._log_handler_ids = _setup_logging(log_config)

        logger.info(f"Starting pool with {self.nodes} nodes ({self.accelerator})")

        # Create event loop in background thread
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="skyward-event-loop",
        )
        self._loop_thread.start()

        # Start pool asynchronously
        try:
            self._run_sync(self._start_async())
            self._active = True
            # Set this pool as active in context
            self._context_token = _active_pool.set(self)
            # Create distributed collections registry
            self._registry = DistributedRegistry()
            _set_active_registry(self._registry)
            logger.info("Pool ready")
        except Exception as e:
            logger.exception(f"Error starting pool: {e}")
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
        try:
            # Reset context
            if self._context_token is not None:
                _active_pool.reset(self._context_token)
                self._context_token = None

            if self._active and self._async_pool:
                self._run_sync_with_timeout(self._async_pool.stop(), timeout=30.0)
        except TimeoutError:
            logger.warning("Pool stop timed out after 30s, forcing cleanup")
        except Exception as e:
            logger.warning(f"Error stopping pool: {e}")
        finally:
            # Cleanup distributed collections registry
            if self._registry is not None:
                self._registry.cleanup()
                _set_active_registry(None)
                self._registry = None

            self._active = False
            self._cleanup()
            logger.info("Pool stopped")

            # Teardown logging at the very end
            if self._log_handler_ids:
                _teardown_logging(self._log_handler_ids)

    def run[T](self, pending: PendingCompute[T]) -> T:
        """Execute a pending computation on the pool.

        Args:
            pending: The lazy computation to execute.

        Returns:
            Result of the computation.
        """
        if not self._active or self._async_pool is None:
            raise RuntimeError("Pool is not active")

        return self._run_sync(
            self._async_pool.run(
                pending.fn,
                *pending.args,
                **pending.kwargs,
            )
        )

    def broadcast[T](self, pending: PendingCompute[T]) -> list[T]:
        """Execute computation on all nodes.

        Args:
            pending: The lazy computation to execute.

        Returns:
            List of results from each node.
        """
        if not self._active or self._async_pool is None:
            raise RuntimeError("Pool is not active")

        return self._run_sync(
            self._async_pool.broadcast(
                pending.fn,
                *pending.args,
                **pending.kwargs,
            )
        )

    def run_parallel(self, group: PendingComputeGroup) -> tuple[Any, ...]:
        """Execute multiple computations in parallel.

        Args:
            group: Group of pending computations.

        Returns:
            Tuple of results in same order as inputs.
        """
        if not self._active or self._async_pool is None:
            raise RuntimeError("Pool is not active")

        async def _run_parallel() -> tuple[Any, ...]:
            tasks = [
                self._async_pool.run(p.fn, *p.args, **p.kwargs)  # type: ignore
                for p in group.items
            ]
            results = await asyncio.gather(*tasks)
            return tuple(results)

        return self._run_sync(_run_parallel())

    def map[T, R](
        self,
        fn: Callable[[T], R],
        items: Sequence[T],
    ) -> list[R]:
        """Map function over items in parallel across nodes.

        Args:
            fn: Function to apply to each item.
            items: Items to process.

        Returns:
            List of results.
        """
        if not self._active or self._async_pool is None:
            raise RuntimeError("Pool is not active")

        async def _map_async() -> list[R]:
            tasks = [
                self._async_pool.run(fn, item)  # type: ignore
                for item in items
            ]
            return list(await asyncio.gather(*tasks))

        return self._run_sync(_map_async())

    # -------------------------------------------------------------------------
    # Distributed Collections
    # -------------------------------------------------------------------------

    def dict(self, name: str, *, consistency: Consistency | None = None) -> DictProxy:
        """Get or create a distributed dict."""
        if self._registry is None:
            raise RuntimeError("Pool is not active")
        return self._registry.dict(name, consistency=consistency)

    def list(self, name: str, *, consistency: Consistency | None = None) -> ListProxy:
        """Get or create a distributed list."""
        if self._registry is None:
            raise RuntimeError("Pool is not active")
        return self._registry.list(name, consistency=consistency)

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
        """Start pool asynchronously."""
        from .module import PoolConfigModule, SkywardModule
        from .orchestrator import InstanceOrchestrator
        from .providers.registry import get_provider_for_config

        # Get provider classes from registry
        # Lazy imports - SDKs are only loaded when actually starting a pool
        handler_cls, module_cls, provider_name = get_provider_for_config(self.provider)
        provider_module = module_cls()

        # Get region from provider config
        region = getattr(self.provider, "region", "unknown")

        # Create pool spec with provider
        spec = PoolSpec(
            nodes=self.nodes,
            accelerator=self.accelerator,
            region=region,
            vcpus=self.vcpus,
            memory_gb=self.memory_gb,
            architecture=self.architecture,
            allocation=self.allocation,
            image=self.image,
            provider=provider_name,  # type: ignore[arg-type]
            max_hourly_cost=self.max_hourly_cost,
        )

        # Build list of modules
        modules = [
            SkywardModule(),
            MonitorModule(),
            provider_module,
            PoolConfigModule(spec=spec, provider_config=self.provider),
        ]

        # Add panel module if enabled
        if self.panel:
            from .observability import PanelModule
            modules.append(PanelModule())

        # Create injector with all modules
        self._injector = Injector(modules)

        # Get instances from injector (auto-wired via @component)
        self._injector.get(InstanceOrchestrator)  # Generic event pipeline
        self._injector.get(handler_cls)            # Provider-specific handler
        self._async_pool = self._injector.get(AsyncPool)
        self._bus = self._injector.get(AsyncEventBus)

        # Initialize panel if enabled (must be after bus is ready)
        if self.panel:
            from .observability import PanelComponent
            self._injector.get(PanelComponent)

        # Start pool
        await self._async_pool.start(timeout=self.timeout)

    def _run_loop(self) -> None:
        """Run event loop in background thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()  # type: ignore

    def _run_sync[T](self, coro: Any) -> T:
        """Run coroutine synchronously."""
        if self._loop is None:
            raise RuntimeError("Event loop not running")

        future: Future[T] = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def _run_sync_with_timeout[T](self, coro: Any, timeout: float) -> T:
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
        return f"SyncComputePool(nodes={self.nodes}, accelerator={self.accelerator}, {status})"


# =============================================================================
# Pool Factory / Decorator
# =============================================================================


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
        timeout: int = 3600,
        panel: bool = True,
        logging: LogConfig | bool = True,
        max_hourly_cost: float | None = None,
    ) -> None:
        self._provider = provider
        self._nodes = nodes
        self._accelerator = accelerator
        self._vcpus = vcpus
        self._memory_gb = memory_gb
        self._architecture = architecture
        self._image = image
        self._allocation = allocation
        self._timeout = timeout
        self._panel = panel
        self._logging = logging
        self._max_hourly_cost = max_hourly_cost
        self._pool: SyncComputePool | None = None

    def _create_pool(self) -> SyncComputePool:
        """Create the underlying pool."""
        provider = self._provider or AWS()
        return SyncComputePool(
            provider=provider,
            nodes=self._nodes,
            accelerator=self._accelerator,
            vcpus=self._vcpus,
            memory_gb=self._memory_gb,
            architecture=self._architecture,  # type: ignore[arg-type]
            allocation=self._allocation,  # type: ignore[arg-type]
            image=self._image or DEFAULT_IMAGE,
            timeout=self._timeout,
            panel=self._panel,
            logging=self._logging,
            max_hourly_cost=self._max_hourly_cost,
        )

    def __enter__(self) -> SyncComputePool:
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

    def __call__[F: Callable[..., Any]](self, fn: F) -> F:
        """Use as decorator."""

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            p = self._create_pool()
            with p:
                return fn(*args, **kwargs)

        return wrapper  # type: ignore


@overload
def pool(
    fn: Callable[..., Any],
) -> Callable[..., Any]: ...


@overload
def pool(
    *,
    provider: Provider | None = None,
    nodes: int = 1,
    accelerator: str | Accelerator | None = None,
    vcpus: int | None = None,
    memory_gb: int | None = None,
    architecture: Literal["x86_64", "arm64"] | None = None,
    image: Image | None = None,
    allocation: Literal["spot", "on-demand", "spot-if-available"] = "spot-if-available",
    timeout: int = 3600,
    panel: bool = True,
    logging: LogConfig | bool = True,
    max_hourly_cost: float | None = None,
) -> _PoolFactory: ...


@overload
def pool(
    provider: Provider | None = None,
    nodes: int = 1,
    accelerator: str | Accelerator | None = None,
    vcpus: int | None = None,
    memory_gb: int | None = None,
    architecture: Literal["x86_64", "arm64"] | None = None,
    image: Image | None = None,
    allocation: Literal["spot", "on-demand", "spot-if-available"] = "spot-if-available",
    timeout: int = 3600,
    panel: bool = True,
    logging: LogConfig | bool = True,
    max_hourly_cost: float | None = None,
) -> _PoolFactory: ...


def pool(
    provider: Provider | Callable[..., Any] | None = None,
    nodes: int = 1,
    accelerator: str | Accelerator | None = None,
    vcpus: int | None = None,
    memory_gb: int | None = None,
    architecture: Literal["x86_64", "arm64"] | None = None,
    image: Image | None = None,
    allocation: Literal["spot", "on-demand", "spot-if-available"] = "spot-if-available",
    timeout: int = 3600,
    panel: bool = True,
    logging: LogConfig | bool = True,
    max_hourly_cost: float | None = None,
) -> _PoolFactory | Callable[..., Any]:
    """Create a compute pool (context manager or decorator).

    Can be used as:

    1. Context manager:
        with pool(provider=AWS(), accelerator="A100") as p:
            result = train(data) >> p

    2. Decorator:
        @pool(provider=AWS(), accelerator="A100")
        def main():
            return train(data) >> sky

    Args:
        provider: Provider configuration (AWS, VastAI, or Verda).
        nodes: Number of nodes.
        accelerator: GPU type.
        image: Environment specification.
        allocation: Instance allocation strategy.
        timeout: Provisioning timeout.
        panel: Enable Rich terminal dashboard (default: True).
        logging: Log configuration. If True, logs to .skyward/skyward.log.
        max_hourly_cost: Maximum hourly cost in USD for the entire cluster.

    Returns:
        A _PoolFactory that works as context manager or decorator.
    """
    # If first arg is callable, it's being used as @pool without parens
    if callable(provider) and not isinstance(provider, (AWS, VastAI, Verda)):
        fn = provider
        factory = _PoolFactory(
            provider=None,
            nodes=nodes,
            accelerator=accelerator,
            vcpus=vcpus,
            memory_gb=memory_gb,
            architecture=architecture,
            image=image,
            allocation=allocation,
            timeout=timeout,
            panel=panel,
            logging=logging,
            max_hourly_cost=max_hourly_cost,
        )
        return factory(fn)

    # Otherwise return factory for context manager or decorator use
    return _PoolFactory(
        provider=provider if isinstance(provider, (AWS, VastAI, Verda)) else None,
        nodes=nodes,
        accelerator=accelerator,
        vcpus=vcpus,
        memory_gb=memory_gb,
        architecture=architecture,
        image=image,
        allocation=allocation,
        timeout=timeout,
        panel=panel,
        logging=logging,
        max_hourly_cost=max_hourly_cost,
    )


# =============================================================================
# Utilities (for use inside @compute functions)
# =============================================================================

# Import from runtime module - this keeps the import chain minimal
# on remote instances (no provider SDK dependencies)
from .runtime import instance_info, shard
from .cluster.info import InstanceInfo


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Core API
    "sky",
    "pool",
    "compute",
    "gather",
    # Types
    "SyncComputePool",
    "PendingCompute",
    "PendingComputeGroup",
    # Utilities
    "InstanceInfo",
    "instance_info",
    "shard",
    # Re-exports for convenience
    "AWS",
    "Image",
]
