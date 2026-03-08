from skyward.actors.autoscaler import _compute_desired
from skyward.actors.messages import PressureReport


def _report(
    queued: int = 0,
    inflight: int = 0,
    total_capacity: int = 4,
    node_count: int = 2,
) -> PressureReport:
    return PressureReport(
        queued=queued,
        inflight=inflight,
        total_capacity=total_capacity,
        node_count=node_count,
    )


class TestComputeDesired:
    def test_zero_nodes_returns_min(self) -> None:
        result = _compute_desired(
            _report(node_count=0),
            current_desired=0, min_nodes=1, max_nodes=8,
            slots_per_node=2, now=100.0, last_busy_time=100.0,
            scale_down_idle_seconds=60.0,
        )
        assert result == 1

    def test_queued_tasks_scale_up(self) -> None:
        result = _compute_desired(
            _report(queued=6, inflight=4, total_capacity=4, node_count=2),
            current_desired=2, min_nodes=1, max_nodes=8,
            slots_per_node=2, now=100.0, last_busy_time=100.0,
            scale_down_idle_seconds=60.0,
        )
        assert result == 5  # 2 existing + ceil(6/2)=3

    def test_queued_tasks_respects_max(self) -> None:
        result = _compute_desired(
            _report(queued=100, inflight=4, total_capacity=4, node_count=2),
            current_desired=2, min_nodes=1, max_nodes=4,
            slots_per_node=2, now=100.0, last_busy_time=100.0,
            scale_down_idle_seconds=60.0,
        )
        assert result == 4

    def test_idle_scales_down_to_min(self) -> None:
        result = _compute_desired(
            _report(queued=0, inflight=0, total_capacity=8, node_count=4),
            current_desired=4, min_nodes=1, max_nodes=8,
            slots_per_node=2, now=200.0, last_busy_time=100.0,
            scale_down_idle_seconds=60.0,
        )
        assert result == 1

    def test_idle_not_long_enough_keeps_current(self) -> None:
        result = _compute_desired(
            _report(queued=0, inflight=0, total_capacity=8, node_count=4),
            current_desired=4, min_nodes=1, max_nodes=8,
            slots_per_node=2, now=130.0, last_busy_time=100.0,
            scale_down_idle_seconds=60.0,
        )
        assert result == 4

    def test_partial_utilization_scale_down(self) -> None:
        result = _compute_desired(
            _report(queued=0, inflight=1, total_capacity=16, node_count=8),
            current_desired=8, min_nodes=1, max_nodes=16,
            slots_per_node=2, now=200.0, last_busy_time=100.0,
            scale_down_idle_seconds=60.0,
        )
        # utilization = 1/16 = 0.0625 < 0.3
        # needed = ceil(1/2) + 1 = 2
        assert result == 2

    def test_steady_state_no_change(self) -> None:
        result = _compute_desired(
            _report(queued=0, inflight=3, total_capacity=4, node_count=2),
            current_desired=2, min_nodes=1, max_nodes=8,
            slots_per_node=2, now=100.0, last_busy_time=100.0,
            scale_down_idle_seconds=60.0,
        )
        assert result == 2

    def test_respects_min_on_scale_down(self) -> None:
        result = _compute_desired(
            _report(queued=0, inflight=0, total_capacity=4, node_count=2),
            current_desired=2, min_nodes=2, max_nodes=8,
            slots_per_node=2, now=200.0, last_busy_time=100.0,
            scale_down_idle_seconds=60.0,
        )
        assert result == 2  # min_nodes is 2

    def test_slots_per_node_zero_handled(self) -> None:
        result = _compute_desired(
            _report(queued=4, inflight=0, total_capacity=2, node_count=2),
            current_desired=2, min_nodes=1, max_nodes=8,
            slots_per_node=0, now=100.0, last_busy_time=100.0,
            scale_down_idle_seconds=60.0,
        )
        assert result == 6  # 2 + ceil(4/1)=4

    def test_deadline_expired_returns_zero(self) -> None:
        result = _compute_desired(
            _report(queued=10, inflight=4, total_capacity=4, node_count=2),
            current_desired=2, min_nodes=1, max_nodes=8,
            slots_per_node=2, now=200.0, last_busy_time=200.0,
            scale_down_idle_seconds=60.0,
            deadline=150.0,  # expired 50s ago
        )
        assert result == 0

    def test_deadline_not_expired_scales_normally(self) -> None:
        result = _compute_desired(
            _report(queued=6, inflight=4, total_capacity=4, node_count=2),
            current_desired=2, min_nodes=1, max_nodes=8,
            slots_per_node=2, now=100.0, last_busy_time=100.0,
            scale_down_idle_seconds=60.0,
            deadline=200.0,  # still 100s left
        )
        assert result == 5  # normal scale-up

    def test_deadline_none_scales_normally(self) -> None:
        result = _compute_desired(
            _report(queued=6, inflight=4, total_capacity=4, node_count=2),
            current_desired=2, min_nodes=1, max_nodes=8,
            slots_per_node=2, now=100.0, last_busy_time=100.0,
            scale_down_idle_seconds=60.0,
            deadline=None,  # no deadline
        )
        assert result == 5
