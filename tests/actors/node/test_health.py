"""Unit tests for the node-actor health-check decision helper."""

from __future__ import annotations

import pytest

from skyward.actors.node.actor import _evaluate_health

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class TestEvaluateHealthHealthy:
    def test_healthy_resets_counter(self) -> None:
        new_count, preempt = _evaluate_health(
            healthy=True, reason=None, current=2, threshold=3,
        )
        assert new_count == 0
        assert preempt is None

    def test_healthy_from_zero_stays_zero(self) -> None:
        new_count, preempt = _evaluate_health(
            healthy=True, reason=None, current=0, threshold=3,
        )
        assert new_count == 0
        assert preempt is None

    def test_healthy_ignores_reason(self) -> None:
        new_count, preempt = _evaluate_health(
            healthy=True, reason="this is ignored", current=5, threshold=3,
        )
        assert new_count == 0
        assert preempt is None


class TestEvaluateHealthFail:
    def test_fail_below_threshold_increments_only(self) -> None:
        new_count, preempt = _evaluate_health(
            healthy=False, reason="GPU gone", current=1, threshold=3,
        )
        assert new_count == 2
        assert preempt is None

    def test_fail_first_check_increments_to_one(self) -> None:
        new_count, preempt = _evaluate_health(
            healthy=False, reason=None, current=0, threshold=3,
        )
        assert new_count == 1
        assert preempt is None

    def test_fail_at_threshold_returns_preempt_reason(self) -> None:
        new_count, preempt = _evaluate_health(
            healthy=False, reason="OOM", current=2, threshold=3,
        )
        assert new_count == 3
        assert preempt is not None
        assert "OOM" in preempt
        assert "3x" in preempt

    def test_fail_above_threshold_still_emits_preempt(self) -> None:
        new_count, preempt = _evaluate_health(
            healthy=False, reason="x", current=4, threshold=3,
        )
        assert new_count == 5
        assert preempt is not None

    def test_fail_with_no_reason_uses_unspecified(self) -> None:
        _, preempt = _evaluate_health(
            healthy=False, reason=None, current=2, threshold=3,
        )
        assert preempt is not None
        assert "unspecified" in preempt

    def test_fail_with_threshold_one_preempts_immediately(self) -> None:
        new_count, preempt = _evaluate_health(
            healthy=False, reason="critical", current=0, threshold=1,
        )
        assert new_count == 1
        assert preempt is not None
        assert "critical" in preempt
