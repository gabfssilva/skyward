from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from skyward.actors.messages import PressureReport
from skyward.actors.node.state import NodeId


def _compute_desired(
    report: PressureReport,
    current_desired: int,
    min_nodes: int,
    max_nodes: int,
    slots_per_node: int,
    deadline: float | None = None,
    now: float | None = None,
) -> int:
    if deadline is not None and now is not None and now > deadline:
        return 0
    if report.node_count == 0:
        return min_nodes
    if report.queued > 0:
        nodes_for_queue = ceil(report.queued / max(slots_per_node, 1))
        return min(max_nodes, max(current_desired, report.node_count + nodes_for_queue))
    return current_desired


@dataclass(frozen=True, slots=True)
class _State:
    desired: int
    last_scale_time: float
    last_pressure: PressureReport | None
    idle: frozenset[NodeId]
    reaping: frozenset[NodeId]
    known_nodes: frozenset[NodeId]
