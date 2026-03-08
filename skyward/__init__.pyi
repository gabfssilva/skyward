"""Skyward — distributed compute orchestration for ML/AI.

    import skyward as sky

    @sky.function
    def train(data):
        return model.fit(data)

    with sky.ComputePool(provider=sky.AWS(), accelerator="A100") as pool:
        result = train(data) >> pool
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from concurrent.futures import Future
from types import TracebackType
from typing import Any, Literal, Self, overload

# ── Sub-module namespaces ─────────────────────────────────────
from skyward import accelerators as accelerators
from skyward import plugins as plugins
from skyward import storage as storage

# ── Private imports (for type annotations only) ───────────────
from skyward.accelerators.spec import Accelerator

# ── Re-exported events (frozen dataclasses, nothing to hide) ──
from skyward.actors.messages import ClusterDestroyed as ClusterDestroyed
from skyward.actors.messages import ClusterId as ClusterId
from skyward.actors.messages import ClusterProvisioned as ClusterProvisioned
from skyward.actors.messages import ClusterReady as ClusterReady
from skyward.actors.messages import ClusterRequested as ClusterRequested
from skyward.actors.messages import Error as Error
from skyward.actors.messages import Event as Event
from skyward.actors.messages import Fact as Fact
from skyward.actors.messages import InstanceBootstrapped as InstanceBootstrapped
from skyward.actors.messages import InstanceDestroyed as InstanceDestroyed
from skyward.actors.messages import InstanceId as InstanceId
from skyward.actors.messages import InstancePreempted as InstancePreempted
from skyward.actors.messages import InstanceProvisioned as InstanceProvisioned
from skyward.actors.messages import InstanceReplaced as InstanceReplaced
from skyward.actors.messages import InstanceRequested as InstanceRequested
from skyward.actors.messages import Log as Log
from skyward.actors.messages import Metric as Metric
from skyward.actors.messages import NodeId as NodeId
from skyward.actors.messages import NodeInstance as NodeInstance
from skyward.actors.messages import ProviderName as ProviderName
from skyward.actors.messages import Request as Request
from skyward.actors.messages import RequestId as RequestId
from skyward.actors.messages import ShutdownCompleted as ShutdownCompleted
from skyward.actors.messages import ShutdownRequested as ShutdownRequested
from skyward.actors.messages import TaskCompleted as TaskCompleted
from skyward.actors.messages import TaskStarted as TaskStarted

# ── Re-exported types (nothing to hide) ───────────────────────
from skyward.api.model import Cluster as Cluster
from skyward.api.model import Instance as Instance
from skyward.api.model import InstanceType as InstanceType
from skyward.api.model import Offer as Offer
from skyward.api.provider import ProviderConfig as ProviderConfig
from skyward.api.runtime import CallbackWriter as CallbackWriter
from skyward.api.runtime import InstanceInfo as InstanceInfo
from skyward.api.runtime import instance_info as instance_info
from skyward.api.runtime import is_head as is_head
from skyward.api.runtime import redirect_output as redirect_output
from skyward.api.runtime import shard as shard
from skyward.api.runtime import silent as silent
from skyward.api.runtime import stderr as stderr
from skyward.api.runtime import stdout as stdout
from skyward.api.spec import DEFAULT_IMAGE as DEFAULT_IMAGE
from skyward.api.spec import AllocationStrategy as AllocationStrategy
from skyward.api.spec import Architecture
from skyward.api.spec import Image as Image
from skyward.api.spec import PipIndex as PipIndex
from skyward.api.spec import PoolSpec as PoolSpec
from skyward.api.spec import SelectionStrategy as SelectionStrategy
from skyward.api.spec import Spec as Spec
from skyward.api.spec import Volume as Volume
from skyward.api.spec import Worker as Worker
from skyward.api.spec import WorkerExecutor as WorkerExecutor
from skyward.distributed import (
    BarrierProxy,
    CounterProxy,
    DictProxy,
    LockProxy,
    QueueProxy,
    SetProxy,
)

# ── Re-exported distributed factories ─────────────────────────
from skyward.distributed import barrier as barrier
from skyward.distributed import counter as counter
from skyward.distributed import dict as dict
from skyward.distributed import lock as lock
from skyward.distributed import queue as queue
from skyward.distributed import set as set
from skyward.distributed.types import Consistency
from skyward.observability import metrics as metrics
from skyward.observability.logging import LogConfig as LogConfig
from skyward.offers.repository import OfferRepository as OfferRepository
from skyward.plugins.plugin import Plugin

# ── Re-exported providers (curated in providers/__init__.pyi) ─
from skyward.providers import AWS as AWS
from skyward.providers import GCP as GCP
from skyward.providers import Container as Container
from skyward.providers import Hyperstack as Hyperstack
from skyward.providers import RunPod as RunPod
from skyward.providers import TensorDock as TensorDock
from skyward.providers import VastAI as VastAI
from skyward.providers import Verda as Verda
from skyward.storage import Storage as Storage

# ── Version ───────────────────────────────────────────────────

__version__: str

# ── Module re-export ──────────────────────────────────────────

from skyward.api import pool as pool

# ── Curated: function decorator ───────────────────────────────

@overload
def function[**P, T](fn: Callable[P, T]) -> Callable[P, PendingFunction[T]]:
    """Mark a function for remote execution (bare decorator)."""
    ...

@overload
def function[**P, T](
    *, timeout: float,
) -> Callable[[Callable[P, T]], Callable[P, PendingFunction[T]]]:
    """Mark a function for remote execution (with timeout)."""
    ...

def function[**P, T](
    fn: Callable[P, T] | None = None,
    *,
    timeout: float | None = None,
) -> Callable[P, PendingFunction[T]] | Callable[[Callable[P, T]], Callable[P, PendingFunction[T]]]:
    """Mark a function for remote execution on a compute pool.

    Wrap a function with ``@sky.function`` to make it return a
    ``PendingFunction[T]`` when called, capturing args without executing.
    Dispatch via operators: ``>> pool``, ``@ pool``, ``> pool``.

    Parameters
    ----------
    fn
        The function to wrap (used when applied without parentheses).
    timeout
        Default per-call timeout in seconds.

    Returns
    -------
    Callable[P, PendingFunction[T]]
        Wrapped function that produces ``PendingFunction`` on each call.

    Examples
    --------
    >>> @sky.function
    ... def train(data):
    ...     return model.fit(data)

    >>> @sky.function(timeout=600)
    ... def long_task(data):
    ...     return heavy_compute(data)
    """
    ...

# ── Curated: gather ───────────────────────────────────────────

def gather(
    *pendings: PendingFunction[Any],
    stream: bool = False,
    ordered: bool = True,
) -> PendingFunctionGroup:
    """Create a parallel execution group from multiple pending functions.

    Alternative to the ``&`` operator with extra control over streaming
    and ordering.

    Parameters
    ----------
    *pendings
        Pending functions to execute in parallel.
    stream
        If ``True``, ``>> pool`` returns a generator yielding results
        as they complete instead of blocking until all finish.
    ordered
        If ``True`` (default), results match input order.
        Only meaningful when ``stream=True``.

    Returns
    -------
    PendingFunctionGroup
        Group ready for dispatch via ``>> pool``.

    Examples
    --------
    >>> a, b, c = sky.gather(task1(), task2(), task3()) >> pool

    >>> for result in sky.gather(task1(), task2(), stream=True) >> pool:
    ...     print(result)
    """
    ...

# ── Curated: offers ───────────────────────────────────────────

async def offers(providers: list[Any]) -> OfferRepository:
    """Load the GPU offer catalog into a queryable repository.

    Parameters
    ----------
    providers
        Provider config instances to fetch offers from.

    Returns
    -------
    OfferRepository
        SQLite-backed queryable catalog.

    Examples
    --------
    >>> repo = await sky.offers([sky.AWS(), sky.VastAI()])
    >>> cheapest = repo.accelerator("A100").spot().cheapest()
    """
    ...

# ── Curated: PendingFunction ─────────────────────────────────

class PendingFunction[T]:
    """Lazy computation — a frozen snapshot of function + args.

    Created by ``@sky.function`` decorated calls. Nothing executes
    until dispatched to a pool via an operator.

    Examples
    --------
    >>> @sky.function
    ... def train(data):
    ...     return model.fit(data)

    >>> pending = train(data)      # PendingFunction — nothing runs yet
    >>> result = pending >> pool    # execute on one node
    >>> results = pending @ pool   # broadcast to all nodes
    >>> future = pending > pool    # async, returns Future[T]
    """

    def with_timeout(self, timeout: float) -> PendingFunction[T]:
        """Return a copy with a per-call timeout override.

        Parameters
        ----------
        timeout
            Timeout in seconds for this specific execution.

        Returns
        -------
        PendingFunction[T]
            New instance with the timeout set.
        """
        ...
    def __rshift__(self, target: Any) -> T:
        """Execute on one node (round-robin) via ``task() >> pool``."""
        ...
    def __gt__(self, target: Any) -> Future[T]:
        """Submit asynchronously via ``task() > pool``."""
        ...
    def __matmul__(self, target: Any) -> list[T] | tuple[T, ...]:
        """Broadcast to all nodes via ``task() @ pool``."""
        ...
    def __and__(
        self, other: PendingFunction[Any] | PendingFunctionGroup,
    ) -> PendingFunctionGroup:
        """Compose parallel group via ``task1() & task2()``."""
        ...

# ── Curated: PendingFunctionGroup ─────────────────────────────

class PendingFunctionGroup:
    """Group of pending functions for parallel execution.

    Created by chaining ``&`` operators or calling ``sky.gather()``.
    Dispatch the group with ``>> pool`` to run all tasks concurrently.

    Examples
    --------
    >>> a, b = (task1() & task2()) >> pool

    >>> results = sky.gather(task1(), task2(), stream=True) >> pool
    >>> for result in results:
    ...     print(result)
    """

    def with_timeout(self, timeout: float) -> PendingFunctionGroup:
        """Return a copy with a group-level timeout override.

        Parameters
        ----------
        timeout
            Timeout in seconds for the entire group execution.
        """
        ...
    def __and__(
        self, other: PendingFunction[Any] | PendingFunctionGroup,
    ) -> PendingFunctionGroup:
        """Extend group via ``(a() & b()) & c()``."""
        ...
    def __rshift__(self, target: Any) -> tuple[Any, ...] | Any:
        """Execute all tasks in parallel via ``group >> pool``."""
        ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Any: ...

# ── Curated: ComputePool ─────────────────────────────────────

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

    >>> with sky.ComputePool(provider=sky.AWS(), accelerator="A100", nodes=4) as pool:
    ...     result = train(data) >> pool
    ...     results = train(data) @ pool
    ...     a, b = (task1() & task2()) >> pool

    Multi-spec with fallback (cheapest across providers):

    >>> with sky.ComputePool(
    ...     sky.Spec(provider=sky.VastAI(), accelerator="A100"),
    ...     sky.Spec(provider=sky.AWS(), accelerator="A100"),
    ...     selection="cheapest",
    ... ) as pool:
    ...     result = train(data) >> pool

    Elastic autoscaling:

    >>> with sky.ComputePool(
    ...     provider=sky.AWS(), accelerator="A100", nodes=(2, 8),
    ... ) as pool:
    ...     result = train(data) >> pool  # scales between 2 and 8 nodes
    """

    @overload
    def __init__(
        self,
        *,
        provider: ProviderConfig,
        nodes: int | tuple[int, int] = ...,
        accelerator: Accelerator | None = ...,
        vcpus: float | None = ...,
        memory_gb: float | None = ...,
        architecture: Architecture | None = ...,
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

    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...

    # ── Execution ─────────────────────────────────────────────

    def run[T](self, pending: PendingFunction[T]) -> T:
        """Execute a pending function on one node (round-robin).

        This is the method behind ``task() >> pool``.

        Parameters
        ----------
        pending
            A ``PendingFunction`` created by calling a ``@sky.function``.

        Returns
        -------
        T
            The remote function's return value.
        """
        ...

    def run_async[T](self, pending: PendingFunction[T]) -> Future[T]:
        """Submit a pending function asynchronously (non-blocking).

        This is the method behind ``task() > pool``.

        Parameters
        ----------
        pending
            A ``PendingFunction`` created by calling a ``@sky.function``.

        Returns
        -------
        Future[T]
            A future that resolves to the remote return value.
        """
        ...

    def broadcast[T](self, pending: PendingFunction[T]) -> list[T]:
        """Execute a pending function on ALL nodes simultaneously.

        This is the method behind ``task() @ pool``.

        Parameters
        ----------
        pending
            A ``PendingFunction`` created by calling a ``@sky.function``.

        Returns
        -------
        list[T]
            One result per node, in node order.
        """
        ...

    def run_parallel(
        self, group: PendingFunctionGroup,
    ) -> tuple[Any, ...] | Generator[Any, None, None]:
        """Execute a group of pending functions concurrently.

        This is the method behind ``(task1() & task2()) >> pool``.

        Parameters
        ----------
        group
            A ``PendingFunctionGroup`` from ``&`` chaining or ``sky.gather()``.

        Returns
        -------
        tuple[Any, ...] | Generator
            Tuple of results if ``stream=False``, generator if ``stream=True``.
        """
        ...

    def map[T, R](
        self,
        fn: Callable[[T], R],
        items: Sequence[T],
    ) -> list[R]:
        """Apply a function to each item, distributing across nodes.

        Each item becomes a separate task dispatched round-robin.
        All results are collected before returning.

        Parameters
        ----------
        fn
            Function to apply to each item.
        items
            Sequence of inputs.

        Returns
        -------
        list[R]
            Results in the same order as ``items``.

        Examples
        --------
        >>> results = pool.map(process, [data1, data2, data3])
        """
        ...

    # ── Distributed collections ───────────────────────────────

    def dict(self, name: str, *, consistency: Consistency | None = None) -> DictProxy:
        """Get or create a distributed dictionary shared across all nodes.

        Parameters
        ----------
        name
            Unique identifier for this collection.
        consistency
            ``"strong"`` or ``"eventual"``. ``None`` uses the system default.

        Returns
        -------
        DictProxy
            Synchronous dict-like proxy backed by the actor system.
        """
        ...

    def set(self, name: str, *, consistency: Consistency | None = None) -> SetProxy:
        """Get or create a distributed set shared across all nodes."""
        ...

    def counter(self, name: str, *, consistency: Consistency | None = None) -> CounterProxy:
        """Get or create a distributed counter shared across all nodes."""
        ...

    def queue(self, name: str) -> QueueProxy:
        """Get or create a distributed FIFO queue shared across all nodes."""
        ...

    def barrier(self, name: str, n: int) -> BarrierProxy:
        """Get or create a distributed barrier.

        Parameters
        ----------
        name
            Unique identifier for this barrier.
        n
            Number of participants that must call ``wait()`` before
            any are released.
        """
        ...

    def lock(self, name: str, timeout: float = 30) -> LockProxy:
        """Get or create a distributed lock.

        Parameters
        ----------
        name
            Unique identifier for this lock.
        timeout
            Default lock acquisition timeout in seconds.
        """
        ...

    # ── Query ─────────────────────────────────────────────────

    def current_nodes(self) -> int:
        """Return the number of nodes currently in the ``ready`` state.

        Returns
        -------
        int
            Count of ready nodes. May be less than requested during
            provisioning or autoscaling.
        """
        ...

    @property
    def concurrency(self) -> int:
        """Number of concurrent task slots per node."""
        ...

    @property
    def is_active(self) -> bool:
        """True if pool is ready for execution."""
        ...

    # ── Named constructor ─────────────────────────────────────

    @classmethod
    def Named(cls, name: str) -> ComputePool:
        """Create a pool from a named configuration in ``skyward.toml``.

        Parameters
        ----------
        name
            Pool name as defined in the ``[pools.<name>]`` section.

        Returns
        -------
        ComputePool
            Pool configured from the TOML section.

        Examples
        --------
        >>> with sky.ComputePool.Named("training") as pool:
        ...     result = train(data) >> pool
        """
        ...

# ── Curated: _Sky singleton ──────────────────────────────────

class _Sky:
    """Singleton that captures ``>>`` and ``@`` operators.

    Use ``sky`` as the right-hand side of operators inside a
    ``with ComputePool(...)`` block. Resolves to the active pool
    via ``ContextVar``.

    Examples
    --------
    >>> with sky.ComputePool(provider=sky.AWS(), accelerator="A100") as pool:
    ...     result = train(data) >> sky
    ...     results = train(data) @ sky
    """

    def __rrshift__(self, pending: Any) -> Any:
        """``pending >> sky`` — execute computation(s)."""
        ...
    def __rmatmul__(self, pending: Any) -> list[Any]:
        """``pending @ sky`` — broadcast to all nodes."""
        ...

sky: _Sky

# ── Curated: App ──────────────────────────────────────────────

class App:
    """Application context manager for console lifecycle and spy wiring.

    Provide a Rich adaptive console and optional spy actor for
    observing pool events. Usually not needed directly — ``ComputePool``
    manages its own ``App`` internally.

    Parameters
    ----------
    console
        Whether to enable Rich console output.

    Examples
    --------
    >>> with sky.App(console=True):
    ...     with sky.ComputePool(...) as pool:
    ...         result = train(data) >> pool
    """

    def __init__(self, *, console: bool = True) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *args: Any) -> None: ...

# ── __all__ ───────────────────────────────────────────────────

__all__ = [
    "__version__",
    "App",
    "sky",
    "pool",
    "function",
    "gather",
    "ComputePool",
    "PendingFunction",
    "PendingFunctionGroup",
    "InstanceInfo",
    "instance_info",
    "shard",
    "stdout",
    "stderr",
    "silent",
    "is_head",
    "CallbackWriter",
    "redirect_output",
    "AWS",
    "Container",
    "GCP",
    "Hyperstack",
    "RunPod",
    "TensorDock",
    "VastAI",
    "Verda",
    "Image",
    "PipIndex",
    "DEFAULT_IMAGE",
    "PoolSpec",
    "AllocationStrategy",
    "InstanceType",
    "Offer",
    "SelectionStrategy",
    "Spec",
    "Volume",
    "Storage",
    "storage",
    "Worker",
    "WorkerExecutor",
    "dict",
    "set",
    "counter",
    "queue",
    "barrier",
    "lock",
    "RequestId",
    "ClusterId",
    "InstanceId",
    "NodeId",
    "ProviderName",
    "NodeInstance",
    "ClusterRequested",
    "InstanceRequested",
    "ShutdownCompleted",
    "ShutdownRequested",
    "ClusterProvisioned",
    "InstanceProvisioned",
    "InstanceBootstrapped",
    "InstancePreempted",
    "InstanceReplaced",
    "InstanceDestroyed",
    "ClusterReady",
    "ClusterDestroyed",
    "TaskStarted",
    "TaskCompleted",
    "Metric",
    "Log",
    "Error",
    "Request",
    "Fact",
    "Event",
    "metrics",
    "LogConfig",
    "accelerators",
    "offers",
    "OfferRepository",
    "plugins",
]
