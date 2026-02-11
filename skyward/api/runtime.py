"""Runtime utilities for distributed execution.

Provides instance info, data sharding, and output control decorators.
This module has NO dependencies on provider SDKs to avoid import errors
when cloudpickle deserializes functions on remote machines.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from functools import wraps
from io import StringIO
from typing import TYPE_CHECKING, Any, Literal, Self, TextIO, TypedDict, overload

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    import numpy.typing as npt
    import torch


class PeerInfo(TypedDict, total=False):
    """Information about a peer node in the compute pool."""

    node: int
    private_ip: str
    public_ip: str | None


class AcceleratorInfo(TypedDict, total=False):
    """Accelerator configuration for a compute pool."""

    type: str
    count: int
    memory_gb: float
    is_trainium: bool


class NetworkInfo(TypedDict, total=False):
    """Network configuration for a compute pool."""

    interface: str
    bandwidth_gbps: float


class InstanceInfo(BaseModel):
    """Information about the current compute pool, parsed from COMPUTE_POOL env var.

    This class provides a typed interface to access pool information
    inside a distributed function.

    Example:
        from skyward import compute_pool

        pool = compute_pool()
        if pool and pool.is_head:
            print(f"I am head node of {pool.total_nodes}")
    """

    node: int = Field(description="Index of this node (0 to total_nodes - 1)")
    worker: int = Field(default=0, description="Worker index within this node (0 to workers_per_node - 1)")
    total_nodes: int = Field(description="Total number of nodes in the pool")
    workers_per_node: int = Field(default=1, description="Number of workers per node (e.g., 2 for MIG 3g.40gb)")
    accelerators: int = Field(description="Number of accelerators on this node")
    total_accelerators: int = Field(description="Total accelerators in the pool")
    head_addr: str = Field(description="IP address of the head node")
    head_port: int = Field(description="Port for head node coordination")
    job_id: str = Field(description="Unique identifier for this pool execution")

    peers: list[PeerInfo] = Field(description="Information about all peers")
    accelerator: AcceleratorInfo | None = Field(default=None, description="Accelerator configuration")
    network: NetworkInfo = Field(description="Network configuration")

    @property
    def global_worker_index(self) -> int:
        """Global index of this worker (0 to total_workers - 1)."""
        return self.node * self.workers_per_node + self.worker

    @property
    def total_workers(self) -> int:
        """Total number of workers across all nodes."""
        return self.total_nodes * self.workers_per_node

    @property
    def is_head(self) -> bool:
        """True if this is the head worker (global_worker_index == 0)."""
        return self.global_worker_index == 0

    @property
    def hostname(self) -> str:
        """Current instance hostname."""
        import socket

        return socket.gethostname()

    @classmethod
    def current(cls) -> Self | None:
        """Get pool info from COMPUTE_POOL environment variable."""
        import os

        cluster_info_json = os.environ.get("COMPUTE_POOL")

        if not cluster_info_json:
            return None

        return cls.model_validate_json(cluster_info_json)


def instance_info() -> InstanceInfo | None:
    """Get information about the current instance.

    Must be called from within a @compute function running on a remote node.

    Returns:
        InstanceInfo with cluster topology, or None if not in a pool.

    Example:
        @compute
        def distributed_task(data):
            info = sky.instance_info()
            if info.is_head:
                print(f"Head node of {info.total_nodes} nodes")
            return process(data)
    """
    return InstanceInfo.current()


def _get_pool_info() -> tuple[int, int]:
    """Get (node, total_nodes) from current pool.

    Returns (0, 1) when not in a pool (local mode).
    """
    pool = instance_info()
    if pool is None:
        return 0, 1
    return pool.node, pool.total_nodes


def _compute_indices(
    n: int,
    node: int,
    total_nodes: int,
    shuffle: bool,
    seed: int,
    drop_last: bool,
) -> list[int]:
    """Compute indices for this node."""
    indices = list(range(n))

    if shuffle:
        import random

        rng = random.Random(seed)
        rng.shuffle(indices)

    if drop_last:
        items_per_node = n // total_nodes
        start = node * items_per_node
        end = start + items_per_node
        return indices[start:end]
    else:
        return indices[node::total_nodes]


def _shard_single(data: Any, indices: list[int]) -> Any:
    """Shard a single array/sequence using precomputed indices."""
    type_name = type(data).__module__ + "." + type(data).__name__

    match type_name:
        case "numpy.ndarray":
            return data[indices]
        case "torch.Tensor":
            import torch

            return data[torch.tensor(indices)]
        case _:
            match data:
                case tuple():
                    return tuple(data[i] for i in indices)
                case _:
                    return [data[i] for i in indices]


@overload
def shard[T](
    data: list[T],
    /,
    *,
    shuffle: bool = False,
    seed: int = 0,
    drop_last: bool = False,
    node: int | None = None,
    total_nodes: int | None = None,
) -> list[T]: ...


@overload
def shard[T](
    data: tuple[T, ...],
    /,
    *,
    shuffle: bool = False,
    seed: int = 0,
    drop_last: bool = False,
    node: int | None = None,
    total_nodes: int | None = None,
) -> tuple[T, ...]: ...


@overload
def shard(
    data: npt.NDArray[Any],
    /,
    *,
    shuffle: bool = False,
    seed: int = 0,
    drop_last: bool = False,
    node: int | None = None,
    total_nodes: int | None = None,
) -> npt.NDArray[Any]: ...


@overload
def shard(
    data: torch.Tensor,
    /,
    *,
    shuffle: bool = False,
    seed: int = 0,
    drop_last: bool = False,
    node: int | None = None,
    total_nodes: int | None = None,
) -> torch.Tensor: ...


@overload
def shard[T1, T2](
    data1: T1,
    data2: T2,
    /,
    *,
    shuffle: bool = False,
    seed: int = 0,
    drop_last: bool = False,
    node: int | None = None,
    total_nodes: int | None = None,
) -> tuple[T1, T2]: ...


@overload
def shard[T1, T2, T3](
    data1: T1,
    data2: T2,
    data3: T3,
    /,
    *,
    shuffle: bool = False,
    seed: int = 0,
    drop_last: bool = False,
    node: int | None = None,
    total_nodes: int | None = None,
) -> tuple[T1, T2, T3]: ...


@overload
def shard[T1, T2, T3, T4](
    data1: T1,
    data2: T2,
    data3: T3,
    data4: T4,
    /,
    *,
    shuffle: bool = False,
    seed: int = 0,
    drop_last: bool = False,
    node: int | None = None,
    total_nodes: int | None = None,
) -> tuple[T1, T2, T3, T4]: ...


@overload
def shard[T1, T2, T3, T4, T5](
    data1: T1,
    data2: T2,
    data3: T3,
    data4: T4,
    data5: T5,
    /,
    *,
    shuffle: bool = False,
    seed: int = 0,
    drop_last: bool = False,
    node: int | None = None,
    total_nodes: int | None = None,
) -> tuple[T1, T2, T3, T4, T5]: ...


def shard(
    *data: Any,
    shuffle: bool = False,
    seed: int = 0,
    drop_last: bool = False,
    node: int | None = None,
    total_nodes: int | None = None,
) -> Any:
    """Shard data across distributed nodes, preserving input type.

    Returns ONLY this node's portion of the data.
    Supports: list, tuple, np.ndarray, torch.Tensor, and any Sequence.

    Can accept multiple arrays at once - they will all be sharded with
    the same indices (useful for keeping x and y aligned).

    Args:
        *data: One or more arrays/sequences to shard.
        shuffle: Shuffle with synchronized seed across all nodes.
        seed: Random seed for reproducible shuffling.
        drop_last: Drop tail items so all nodes get equal count.
        node: Override node index (for testing).
        total_nodes: Override total_nodes (for testing).

    Returns:
        If single argument: This node's shard with same type as input.
        If multiple arguments: Tuple of shards.

    Example:
        # Single array
        my_data = shard(full_dataset, shuffle=True, seed=42)

        # Multiple arrays (keeps alignment)
        x_train, y_train = shard(x_train, y_train)

        # Four arrays at once
        x_train, y_train, x_test, y_test = shard(x_train, y_train, x_test, y_test)
    """
    if not data:
        raise ValueError("shard() requires at least one argument")

    if node is None or total_nodes is None:
        auto_node, auto_total_nodes = _get_pool_info()
        node = node if node is not None else auto_node
        total_nodes = total_nodes if total_nodes is not None else auto_total_nodes

    if len(data) == 1:
        indices = _compute_indices(len(data[0]), node, total_nodes, shuffle, seed, drop_last)
        return _shard_single(data[0], indices)

    def shard_one(d: Any) -> Any:
        indices = _compute_indices(len(d), node, total_nodes, shuffle, seed, drop_last)
        return _shard_single(d, indices)

    return tuple(shard_one(d) for d in data)


type OutputPredicate = Callable[["InstanceInfo"], bool]
"""Predicate that determines if a worker should emit output."""

type OutputSpec = OutputPredicate | Literal["head"]
"""Either a predicate or 'head' shortcut for the most common pattern."""


def is_head(info: InstanceInfo) -> bool:
    """True if this is the head worker (node == 0)."""
    return info.is_head


def _resolve_predicate(spec: OutputSpec) -> OutputPredicate:
    """Convert an OutputSpec to a predicate function."""
    match spec:
        case "head":
            return is_head
        case _ if callable(spec):
            return spec
        case _:
            raise ValueError(f"Invalid output spec: {spec!r}")


def stdout[**P, R](
    only: OutputSpec,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Control stdout emission in distributed execution.

    Silences stdout for workers that don't match the predicate.
    stderr is NOT affected - errors from any worker are always visible.

    Args:
        only: Predicate or "head" shortcut. Workers matching this emit stdout.
            - "head": Only head worker (node == 0)
            - Callable[[InstanceInfo], bool]: Custom predicate

    Returns:
        Decorator that wraps the function with stdout control.
    """
    predicate = _resolve_predicate(only)

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            info = instance_info()

            if info is None or predicate(info):
                return fn(*args, **kwargs)

            with redirect_stdout(StringIO()):
                return fn(*args, **kwargs)

        return wrapper

    return decorator


def stderr[**P, R](
    only: OutputSpec,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Control stderr emission in distributed execution.

    Silences stderr for workers that don't match the predicate.
    Use with caution - silencing errors can hide problems.

    Args:
        only: Predicate or "head" shortcut. Workers matching this emit stderr.
    """
    predicate = _resolve_predicate(only)

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            info = instance_info()

            if info is None or predicate(info):
                return fn(*args, **kwargs)

            with redirect_stderr(StringIO()):
                return fn(*args, **kwargs)

        return wrapper

    return decorator


def silent[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """Silence both stdout and stderr completely.

    Useful for functions that should never emit output regardless of rank.
    """

    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            return fn(*args, **kwargs)

    return wrapper


class CallbackWriter(TextIO):
    def __init__(self, callback: Callable[[str], None]) -> None:
        self._callback = callback
        self._buffer = StringIO()

    def write(self, s: str) -> int:
        self._callback(s)
        return self._buffer.write(s)

    def getvalue(self) -> str:
        return self._buffer.getvalue()

    def read(self, n: int = -1) -> str:
        return self._buffer.read(n)

    def readline(self, limit: int = -1) -> str:
        return self._buffer.readline(limit)

    def flush(self) -> None:
        self._buffer.flush()

    def close(self) -> None:
        self._buffer.close()

    def seekable(self) -> bool:
        return False

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return True


@contextmanager
def redirect_output(
    callback: Callable[[str], None],
) -> Generator[tuple[CallbackWriter, CallbackWriter]]:
    out = CallbackWriter(callback)
    err = CallbackWriter(callback)
    with redirect_stdout(out), redirect_stderr(err):  # type: ignore[type-var]
        yield out, err
