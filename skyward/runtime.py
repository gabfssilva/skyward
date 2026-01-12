"""Lightweight runtime utilities for v2.

This module provides functions that run on remote instances during execution.
It has NO dependencies on provider SDKs (aioboto3, etc.) to avoid import errors
when cloudpickle deserializes functions on remote machines.

These functions are re-exported from skyward for user convenience, but
the actual implementation lives here to keep the import chain minimal.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Self


@dataclass
class PoolInfo:
    """Information about the current instance in a compute pool.

    Provides cluster topology info needed for distributed training coordination.

    Attributes:
        node: Node index (0-based).
        total_nodes: Total number of nodes in pool.
        hostname: Instance hostname.
        head_addr: IP address of head node (node 0) for coordination.
        head_port: Port for distributed coordination (default 29500).
    """

    node: int
    total_nodes: int
    hostname: str
    head_addr: str = "127.0.0.1"
    head_port: int = 29500

    @property
    def is_head(self) -> bool:
        """True if this is the head node (node 0)."""
        return self.node == 0

    @classmethod
    def current(cls) -> Self | None:
        """Get instance info from environment variables.

        Returns:
            PoolInfo if running in a compute pool, None otherwise.
        """
        node_id = os.environ.get("SKYWARD_NODE_ID")
        if node_id is None:
            return None

        return cls(
            node=int(node_id),
            total_nodes=int(os.environ.get("SKYWARD_TOTAL_NODES", "1")),
            hostname=socket.gethostname(),
            head_addr=os.environ.get("SKYWARD_HEAD_ADDR", "127.0.0.1"),
            head_port=int(os.environ.get("SKYWARD_HEAD_PORT", "29500")),
        )


def instance_info() -> PoolInfo | None:
    """Get information about the current instance.

    Must be called from within a @compute function running on a remote node.

    Returns:
        PoolInfo with cluster topology, or None if not in a pool.

    Example:
        @compute
        def distributed_task(data):
            info = sky.instance_info()
            if info.is_head:
                print(f"Head node of {info.total_nodes} nodes")
            return process(data)
    """
    return PoolInfo.current()


def shard[T: Sequence[Any]](
    *arrays: T,
    shuffle: bool = False,
    seed: int | None = None,
) -> tuple[T, ...] | T:
    """Shard arrays for the current node.

    Automatically divides data so each node gets its portion.
    All nodes must call shard() with the same data and parameters.

    Args:
        *arrays: One or more sequences to shard.
        shuffle: Whether to shuffle before sharding.
        seed: Random seed for reproducible shuffling.

    Returns:
        Sharded portion for this node. Returns tuple if multiple arrays,
        single array if one array.

    Example:
        @compute
        def train(full_x, full_y):
            # Each node gets different portion
            x, y = sky.shard(full_x, full_y, shuffle=True, seed=42)
            return train_model(x, y)
    """
    import random

    info = instance_info()
    node_id = info.node if info else 0
    total_nodes = info.total_nodes if info else 1

    def compute_indices(n: int) -> list[int]:
        """Compute indices for this node's shard."""
        indices = list(range(n))

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(indices)

        # Calculate shard boundaries
        per_node = n // total_nodes
        remainder = n % total_nodes

        # Distribute remainder across first nodes
        start = node_id * per_node + min(node_id, remainder)
        end = start + per_node + (1 if node_id < remainder else 0)

        return indices[start:end]

    def shard_single(data: T, indices: list[int]) -> T:
        """Shard a single array/sequence using precomputed indices."""
        # Check type by module + name to avoid importing numpy/torch
        type_name = type(data).__module__ + "." + type(data).__name__

        # numpy.ndarray - use fancy indexing
        if type_name == "numpy.ndarray":
            return data[indices]  # type: ignore

        # torch.Tensor - use index_select
        if type_name == "torch.Tensor":
            import torch
            return data[torch.tensor(indices)]  # type: ignore

        # tuple - return tuple
        if isinstance(data, tuple):
            return tuple(data[i] for i in indices)  # type: ignore

        # list or other Sequence - return list
        return [data[i] for i in indices]  # type: ignore

    # Single argument: return directly (preserves type)
    if len(arrays) == 1:
        indices = compute_indices(len(arrays[0]))
        return shard_single(arrays[0], indices)

    # Multiple arguments: shard each with same indices (handles different sizes)
    def shard_one(arr: T) -> T:
        indices = compute_indices(len(arr))
        return shard_single(arr, indices)

    return tuple(shard_one(arr) for arr in arrays)


__all__ = [
    "PoolInfo",
    "instance_info",
    "shard",
]
