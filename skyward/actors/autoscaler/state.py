from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from skyward.actors.messages import PressureReport


def _compute_desired(
    report: PressureReport,
    current_desired: int,
    min_nodes: int,
    max_nodes: int,
    slots_per_node: int,
    now: float,
    last_busy_time: float,
    scale_down_idle_seconds: float,
    deadline: float | None = None,
) -> int:
    if deadline is not None and now > deadline:
        return 0

    if report.node_count == 0:
        return min_nodes

    if report.queued > 0:
        nodes_for_queue = ceil(report.queued / max(slots_per_node, 1))
        return min(max_nodes, max(current_desired, report.node_count + nodes_for_queue))

    if report.inflight == 0 and now - last_busy_time > scale_down_idle_seconds:
        return min_nodes

    if report.queued == 0 and report.total_capacity > 0:
        utilization = report.inflight / report.total_capacity
        if utilization < 0.3 and now - last_busy_time > scale_down_idle_seconds:
            needed = ceil(report.inflight / max(slots_per_node, 1)) + 1
            return min(current_desired, max(min_nodes, needed))

    return current_desired


@dataclass(frozen=True, slots=True)
class _State:
    desired: int
    last_scale_time: float
    last_pressure: PressureReport | None
    last_busy_time: float
