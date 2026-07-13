"""Value objects exchanged between control-plane components.

Not part of the domain vocabulary (that lives in ``skyward.api.facts``):
these are the handful of records the pool, node, task manager, and
autoscaler pass to one another.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["HeadAddressKnown", "NodeInterruptedError", "PressureReport"]


class NodeInterruptedError(Exception):
    """Task lost to infrastructure failure (preemption, connection loss)."""

    def __init__(self, node_id: int, reason: str) -> None:
        self.node_id = node_id
        self.reason = reason
        super().__init__(f"Node {node_id} interrupted: {reason}")


@dataclass(frozen=True, slots=True)
class HeadAddressKnown:
    """Node → pool: where the casty seed lives and how workers are sized."""

    head_addr: str
    casty_port: int
    num_nodes: int
    worker_concurrency: int
    worker_executor: str
    worker_reuse_processes: bool = True


@dataclass(frozen=True, slots=True)
class PressureReport:
    """Task manager → autoscaler: how loaded the pool is right now."""

    queued: int
    inflight: int
    total_capacity: int
    node_count: int
