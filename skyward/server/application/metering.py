"""What a compute has cost so far, said out loud every few seconds.

Cost is not accumulated, it is derived: every node carries the price it was
bought at, when it was launched and when it was given back, so the total at any
instant is a sum over rows. There is no counter to drift, nothing to persist,
and a daemon restart changes nothing — the meter is a reader.
"""

from __future__ import annotations

from skyward.server.persistence.computes import ComputeStore
from skyward.server.persistence.events import EventStore
from skyward.server.persistence.nodes import NodeStore
from skyward.server.persistence.store import now
from skyward.shared.billing import accrued
from skyward.shared.events import CostEvent
from skyward.shared.observability import logger

logger = logger.bind(component="metering")


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
            cost = round(sum(accrued(node, at) for node in metered), 6)
            billing = sum(1 for node in metered if node.terminated_at is None)
            logger.bind(compute_id=compute_id).debug("{} machines billing, ${:.4f} so far", billing, cost)
            await self._events.publish(CostEvent(compute=compute_id, cost=cost, nodes=billing, at=at))
