from __future__ import annotations

from math import ceil

from skyward.actors.messages import PressureReport


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
