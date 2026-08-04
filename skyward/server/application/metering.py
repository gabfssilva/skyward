"""What a compute has cost so far, said out loud every few seconds.

Cost is not accumulated, it is derived: every node carries the price it was
bought at, when it was launched and when it was given back, so the total at any
instant is a sum over rows. There is no counter to drift, nothing to persist,
and a daemon restart changes nothing — the meter is a reader.
"""

from __future__ import annotations

from datetime import datetime
from math import ceil

from skyward.server.persistence.computes import ComputeStore
from skyward.server.persistence.events import EventStore
from skyward.server.persistence.nodes import NodeStore
from skyward.server.persistence.store import now
from skyward.shared import codec
from skyward.shared.schemas import BillingUnit, CostEvent, Event, Node

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


class Meter:
    """Publishes each live compute's accrued cost as a gauge.

    Published, not recorded: a reading every few seconds has no replay value and
    the event table has no GC to save it from one. A subscriber that missed a
    sample gets a fresher one moments later, and the truth is derivable from the
    node rows at any time anyway.
    """

    def __init__(self, computes: ComputeStore, nodes: NodeStore, events: EventStore) -> None:
        self._computes = computes
        self._nodes = nodes
        self._events = events

    async def sample(self) -> None:
        at = now()
        for compute_id in await self._computes.live():
            nodes = await self._nodes.of(compute_id)
            metered = [node for node in nodes if node.launched_at is not None]
            payload = await codec.json(Event).encode(
                CostEvent(
                    compute=compute_id,
                    cost=round(sum(accrued(node, at) for node in metered), 6),
                    nodes=sum(1 for node in metered if node.terminated_at is None),
                    at=at,
                )
            )
            await self._events.publish("compute.cost", payload, compute=compute_id)
