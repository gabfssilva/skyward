"""What holding a machine costs.

One rule, on the floor because both sides of the store read it: the meter publishes
a live compute's running total from it, and a deleted compute's bill is summed with
it whenever the compute is read. Nothing accumulates a total anywhere — every node
carries the price it was bought at, when it was launched and when it was given back,
so what a compute has cost at any instant is a sum over its rows.
"""

from __future__ import annotations

from datetime import datetime
from math import ceil

from skyward.shared.schemas import BillingUnit, Node

UNIT_SECONDS: dict[BillingUnit, int] = {"second": 1, "minute": 60, "hour": 3600}


def accrued(node: Node, at: datetime) -> float:
    """What this node has cost up to ``at``, in dollars.

    Elapsed time is rounded up to the provider's billing unit — a machine held
    for 61 seconds is billed 2 minutes on a per-minute provider and a full hour
    on a per-hour one. A node that never got a machine, or whose price was never
    learned, costs nothing that can be counted; it contributes zero rather than
    a guess.
    """
    if node.launched_at is None or node.price_per_hour is None:
        return 0.0

    unit = UNIT_SECONDS[node.billing_unit or "hour"]
    elapsed = ((node.terminated_at or at) - node.launched_at).total_seconds()
    billed = ceil(max(elapsed, 0.0) / unit) * unit
    return billed / 3600 * node.price_per_hour


__all__ = ["UNIT_SECONDS", "accrued"]
