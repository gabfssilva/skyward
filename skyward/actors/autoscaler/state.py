from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import ceil

from skyward.actors.messages import BoundsChanged, PressureReport
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
    if report.queued > 0:
        nodes_for_queue = ceil(report.queued / max(slots_per_node, 1))
        return min(max_nodes, max(current_desired, report.node_count + nodes_for_queue))
    if report.node_count == 0:
        return min_nodes
    return current_desired


@dataclass(frozen=True, slots=True)
class _State:
    desired: int
    last_scale_time: float
    last_pressure: PressureReport | None
    min_nodes: int
    max_nodes: int
    idle: frozenset[NodeId] = field(default_factory=frozenset)
    reaping: frozenset[NodeId] = field(default_factory=frozenset)
    known_nodes: frozenset[NodeId] = field(default_factory=frozenset)


def _apply_bounds(s: _State, msg: BoundsChanged) -> _State:
    clamped = max(msg.min, min(msg.desired, msg.max))
    return replace(s, min_nodes=msg.min, max_nodes=msg.max, desired=clamped)
