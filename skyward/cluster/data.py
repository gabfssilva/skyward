"""Dataset sharding utilities for distributed training."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, TypeVar, overload

if TYPE_CHECKING:
    import numpy.typing as npt
    import torch

__all__ = ["shard", "shard_iterator", "DistributedSampler"]

T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
T4 = TypeVar("T4")
T5 = TypeVar("T5")


def _get_pool_info() -> tuple[int, int]:
    """Get (node, total_nodes) from current pool.

    Returns (0, 1) when not in a pool (local mode).
    """
    from skyward.cluster.info import instance_info

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
    # Preserve type based on input
    type_name = type(data).__module__ + "." + type(data).__name__

    # numpy.ndarray - use fancy indexing
    if type_name == "numpy.ndarray":
        return data[indices]

    # torch.Tensor - use index_select
    if type_name == "torch.Tensor":
        import torch

        return data[torch.tensor(indices)]

    # tuple - return tuple
    if isinstance(data, tuple):
        return tuple(data[i] for i in indices)

    # list or other Sequence - return list
    return [data[i] for i in indices]


# ============= Overloads para shard() com múltiplos argumentos =============


# Single argument overloads - ordem: mais específico primeiro
# list e tuple antes de tipos opcionais como ndarray/Tensor
@overload
def shard(
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
def shard(
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


# Multiple argument overloads (retornam tupla)
@overload
def shard(
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
def shard(
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
def shard(
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
def shard(
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

    # Single argument: return directly (preserves type)
    if len(data) == 1:
        indices = _compute_indices(len(data[0]), node, total_nodes, shuffle, seed, drop_last)
        return _shard_single(data[0], indices)

    # Multiple arguments: compute indices per array (handles different sizes)
    def shard_one(d: Any) -> Any:
        indices = _compute_indices(len(d), node, total_nodes, shuffle, seed, drop_last)
        return _shard_single(d, indices)

    return tuple(shard_one(d) for d in data)


# ============= shard_iterator =============


def shard_iterator(
    iterator: Iterator[T],
    *,
    node: int | None = None,
    total_nodes: int | None = None,
) -> Iterator[T]:
    """Shard an iterator across distributed nodes (round-robin).

    Yields ONLY items belonging to this node.

    Args:
        iterator: Any iterator or generator.
        node: Override node index (for testing).
        total_nodes: Override total_nodes (for testing).

    Yields:
        Items for this node (every total_nodes-th item starting at node).
    """
    if node is None or total_nodes is None:
        auto_node, auto_total_nodes = _get_pool_info()
        node = node if node is not None else auto_node
        total_nodes = total_nodes if total_nodes is not None else auto_total_nodes

    for i, item in enumerate(iterator):
        if i % total_nodes == node:
            yield item


# ============= DistributedSampler =============


class DistributedSampler(Sequence[int]):
    """PyTorch DataLoader compatible sampler for distributed training.

    Returns ONLY indices for this node.
    Call set_epoch() before each epoch to update shuffle.

    Example:
        sampler = DistributedSampler(dataset, shuffle=True)
        loader = DataLoader(dataset, sampler=sampler)

        for epoch in range(epochs):
            sampler.set_epoch(epoch)
            for batch in loader:
                ...
    """

    def __init__(
        self,
        dataset: Sequence[Any],
        *,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        node: int | None = None,
        total_nodes: int | None = None,
    ) -> None:
        if node is None or total_nodes is None:
            auto_node, auto_total_nodes = _get_pool_info()
            node = node if node is not None else auto_node
            total_nodes = total_nodes if total_nodes is not None else auto_total_nodes

        self.dataset = dataset
        self.node = node
        self.total_nodes = total_nodes
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self._indices: list[int] | None = None

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic shuffling."""
        self.epoch = epoch
        self._indices = None

    @property
    def indices(self) -> list[int]:
        """Get indices for this node (cached)."""
        if self._indices is None:
            self._indices = _compute_indices(
                len(self.dataset),
                self.node,
                self.total_nodes,
                self.shuffle,
                self.seed + self.epoch,
                self.drop_last,
            )
        return self._indices

    def __len__(self) -> int:
        return len(self.indices)

    @overload
    def __getitem__(self, idx: int) -> int: ...

    @overload
    def __getitem__(self, idx: slice) -> Sequence[int]: ...

    def __getitem__(self, idx: int | slice) -> int | Sequence[int]:
        return self.indices[idx]

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)
