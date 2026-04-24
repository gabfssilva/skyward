from skyward.actors.autoscaler.state import _apply_bounds, _compute_desired, _State
from skyward.actors.messages import BoundsChanged, PressureReport


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
            slots_per_node=2,
        )
        assert result == 1

    def test_queued_tasks_scale_up(self) -> None:
        result = _compute_desired(
            _report(queued=6, inflight=4, total_capacity=4, node_count=2),
            current_desired=2, min_nodes=1, max_nodes=8,
            slots_per_node=2,
        )
        assert result == 5  # 2 existing + ceil(6/2)=3

    def test_queued_tasks_respects_max(self) -> None:
        result = _compute_desired(
            _report(queued=100, inflight=4, total_capacity=4, node_count=2),
            current_desired=2, min_nodes=1, max_nodes=4,
            slots_per_node=2,
        )
        assert result == 4

    def test_steady_state_no_change(self) -> None:
        result = _compute_desired(
            _report(queued=0, inflight=3, total_capacity=4, node_count=2),
            current_desired=2, min_nodes=1, max_nodes=8,
            slots_per_node=2,
        )
        assert result == 2

    def test_slots_per_node_zero_handled(self) -> None:
        result = _compute_desired(
            _report(queued=4, inflight=0, total_capacity=2, node_count=2),
            current_desired=2, min_nodes=1, max_nodes=8,
            slots_per_node=0,
        )
        assert result == 6  # 2 + ceil(4/1)=4

    def test_deadline_expired_returns_zero(self) -> None:
        result = _compute_desired(
            _report(queued=10, inflight=4, total_capacity=4, node_count=2),
            current_desired=2, min_nodes=1, max_nodes=8,
            slots_per_node=2,
            deadline=150.0, now=200.0,  # expired 50s ago
        )
        assert result == 0

    def test_deadline_not_expired_scales_normally(self) -> None:
        result = _compute_desired(
            _report(queued=6, inflight=4, total_capacity=4, node_count=2),
            current_desired=2, min_nodes=1, max_nodes=8,
            slots_per_node=2,
            deadline=200.0, now=100.0,  # still 100s left
        )
        assert result == 5  # normal scale-up

    def test_deadline_none_scales_normally(self) -> None:
        result = _compute_desired(
            _report(queued=6, inflight=4, total_capacity=4, node_count=2),
            current_desired=2, min_nodes=1, max_nodes=8,
            slots_per_node=2,
            deadline=None,  # no deadline
        )
        assert result == 5


class TestApplyBounds:
    """The reducer _apply_bounds(state, msg) is the purest way to test BoundsChanged."""

    def _state(self, desired: int, min_n: int, max_n: int) -> _State:
        return _State(
            desired=desired,
            last_scale_time=0.0,
            last_pressure=None,
            min_nodes=min_n,
            max_nodes=max_n,
        )

    def test_new_bounds_stored(self) -> None:
        s = self._state(desired=4, min_n=2, max_n=8)
        new_s = _apply_bounds(s, BoundsChanged(min=1, max=6, desired=3))
        assert new_s.min_nodes == 1
        assert new_s.max_nodes == 6

    def test_desired_rebased_to_message(self) -> None:
        s = self._state(desired=4, min_n=2, max_n=8)
        new_s = _apply_bounds(s, BoundsChanged(min=1, max=6, desired=3))
        assert new_s.desired == 3

    def test_desired_clamped_above_new_max(self) -> None:
        s = self._state(desired=8, min_n=2, max_n=8)
        new_s = _apply_bounds(s, BoundsChanged(min=1, max=4, desired=10))
        assert new_s.desired == 4

    def test_desired_clamped_below_new_min(self) -> None:
        s = self._state(desired=1, min_n=1, max_n=8)
        new_s = _apply_bounds(s, BoundsChanged(min=3, max=8, desired=1))
        assert new_s.desired == 3

    def test_no_op_transition(self) -> None:
        """BoundsChanged with same values should produce state equal to input."""
        s = self._state(desired=4, min_n=2, max_n=8)
        new_s = _apply_bounds(s, BoundsChanged(min=2, max=8, desired=4))
        assert new_s.min_nodes == 2
        assert new_s.max_nodes == 8
        assert new_s.desired == 4

    def test_degenerate_band_min_equals_max_equals_desired(self) -> None:
        """Fixed-pool shape (min==max==desired) should be a stable fixed point."""
        s = self._state(desired=3, min_n=3, max_n=3)
        new_s = _apply_bounds(s, BoundsChanged(min=3, max=3, desired=3))
        assert (new_s.min_nodes, new_s.max_nodes, new_s.desired) == (3, 3, 3)


class TestComputeDesiredRespectsMutableBounds:
    """After BoundsChanged shrinks max, compute_desired must not produce values > new max."""

    def test_queued_work_capped_by_new_max(self) -> None:
        report = PressureReport(queued=50, inflight=0, total_capacity=2, node_count=1)
        # Simulate post-BoundsChanged: max shrunk from 16 → 3.
        result = _compute_desired(
            report,
            current_desired=1, min_nodes=1, max_nodes=3,
            slots_per_node=1,
        )
        assert result == 3
