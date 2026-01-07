"""Compute module - lazy computation primitives.

Contains PendingCompute, gather, and compute decorator.
"""

from skyward.compute.pending import (
    PendingBatch,
    PendingCompute,
    PoolTarget,
    compute,
    gather,
)

__all__ = [
    "PendingCompute",
    "PendingBatch",
    "PoolTarget",
    "compute",
    "gather",
]
