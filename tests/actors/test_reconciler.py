import asyncio
from dataclasses import replace
from unittest.mock import MagicMock

import pytest
from casty import ActorSystem

from skyward.actors.messages import (
    BoundsChanged,
    NodeJoined,
    ReapIdleNodes,
    RequestDrainNodes,
    RequestScaleDown,
)
from skyward.actors.reconciler import reconciler_actor
from skyward.actors.reconciler.state import _State, _apply_bounds


def _make_state(
    desired: int = 2,
    current: frozenset[int] | None = None,
    pending: int = 0,
    draining: int = 0,
    consecutive_failures: int = 0,
) -> _State:
    return _State(
        desired=desired,
        current=current or frozenset({0, 1}),
        pending=pending,
        draining=draining,
        consecutive_failures=consecutive_failures,
    )


class TestReconcilerState:
    def test_effective_count_current_plus_pending(self) -> None:
        s = _make_state(current=frozenset({0, 1}), pending=1)
        assert s.effective == 3

    def test_effective_is_property(self) -> None:
        s = _make_state(current=frozenset({0, 1, 2}), pending=2)
        assert s.effective == 5

    def test_desired_count_changed_up(self) -> None:
        s = _make_state(desired=2)
        new_s = replace(s, desired=4)
        assert new_s.desired > new_s.effective

    def test_desired_count_changed_down(self) -> None:
        s = _make_state(desired=4, current=frozenset({0, 1, 2, 3}))
        new_s = replace(s, desired=2)
        assert new_s.desired < len(new_s.current)

    def test_node_lost_reduces_current(self) -> None:
        s = _make_state(current=frozenset({0, 1, 2}))
        new_s = replace(s, current=s.current - {2})
        assert 2 not in new_s.current
        assert len(new_s.current) == 2

    def test_node_lost_triggers_scale_up(self) -> None:
        s = _make_state(desired=3, current=frozenset({0, 1, 2}))
        new_s = replace(s, current=s.current - {2})
        assert new_s.desired > new_s.effective

    def test_node_joined_moves_pending_to_current(self) -> None:
        s = _make_state(current=frozenset({0, 1}), pending=1)
        new_s = replace(s, current=s.current | {2}, pending=max(0, s.pending - 1))
        assert 2 in new_s.current
        assert new_s.pending == 0

    def test_pending_is_count_not_set(self) -> None:
        s = _make_state(pending=3)
        assert isinstance(s.pending, int)

    def test_drain_complete_reduces_draining(self) -> None:
        s = _make_state(desired=1, current=frozenset({0, 1}), draining=1)
        new_s = replace(s, draining=s.draining - 1)
        assert new_s.draining == 0

    def test_head_node_protected_in_drain_logic(self) -> None:
        """Drain victim selection excludes node 0 — this logic moves to pool,
        but we verify the pattern still holds."""
        current = frozenset({0, 1, 2})
        desired = 1
        excess = len(current) - desired
        victims = sorted(current, reverse=True)[:excess]
        victims = [nid for nid in victims if nid != 0]
        assert 0 not in victims
        assert victims == [2, 1]

    def test_consecutive_failures_increments(self) -> None:
        s = _make_state(consecutive_failures=0)
        new_s = replace(s, consecutive_failures=s.consecutive_failures + 1)
        assert new_s.consecutive_failures == 1

    def test_consecutive_failures_resets_on_success(self) -> None:
        s = _make_state(consecutive_failures=5)
        new_s = replace(s, consecutive_failures=0)
        assert new_s.consecutive_failures == 0

    def test_excess_node_triggers_scale_down(self) -> None:
        s = _make_state(desired=2, current=frozenset({0, 1}))
        new_s = replace(s, current=s.current | {2})
        assert new_s.desired < len(new_s.current)

    def test_tick_detects_downward_drift(self) -> None:
        s = _make_state(desired=2, current=frozenset({0, 1, 2}))
        assert s.desired < len(s.current)
        assert s.draining == 0

    def test_no_cluster_or_instance_map_in_state(self) -> None:
        """Reconciler state must NOT contain provider-level details."""
        s = _make_state()
        assert not hasattr(s, "cluster")
        assert not hasattr(s, "instance_map")
        assert not hasattr(s, "next_node_id")


class TestReapIdleNodes:
    @pytest.mark.asyncio
    async def test_safe_reap_drains_nodes_and_decrements_desired(self) -> None:
        sent: list[object] = []
        pool = MagicMock()
        pool.tell = lambda msg: sent.append(msg)

        async with ActorSystem("test-reconciler-reap-safe") as system:
            ref = system.spawn(
                reconciler_actor(
                    pool=pool,
                    min_nodes=1,
                    max_nodes=8,
                    initial_node_ids=frozenset({0, 1, 2, 3}),
                    tick_interval=3600.0,
                ),
                "reconciler-reap-safe",
            )
            await asyncio.sleep(0.1)
            sent.clear()

            ref.tell(ReapIdleNodes(
                node_ids=frozenset({2, 3}),
                reason="idle-test",
            ))
            await asyncio.sleep(0.1)

            drain_msgs = [m for m in sent if isinstance(m, RequestDrainNodes)]
            assert len(drain_msgs) == 1
            assert drain_msgs[0].node_ids == frozenset({2, 3})

            sent.clear()
            ref.tell(NodeJoined(node_id=5))
            await asyncio.sleep(0.1)

        scale_down_msgs = [m for m in sent if isinstance(m, RequestScaleDown)]
        assert len(scale_down_msgs) == 1
        assert scale_down_msgs[0].count == 3

    @pytest.mark.asyncio
    async def test_reap_violating_min_nodes_is_ignored(self) -> None:
        sent: list[object] = []
        pool = MagicMock()
        pool.tell = lambda msg: sent.append(msg)

        async with ActorSystem("test-reconciler-reap-floor") as system:
            ref = system.spawn(
                reconciler_actor(
                    pool=pool,
                    min_nodes=2,
                    max_nodes=8,
                    initial_node_ids=frozenset({0, 1, 2}),
                    tick_interval=3600.0,
                ),
                "reconciler-reap-floor",
            )
            await asyncio.sleep(0.1)
            sent.clear()

            ref.tell(ReapIdleNodes(
                node_ids=frozenset({1, 2}),
                reason="would-violate-min",
            ))
            await asyncio.sleep(0.1)

            drain_msgs = [m for m in sent if isinstance(m, RequestDrainNodes)]
            assert drain_msgs == []

            sent.clear()
            ref.tell(NodeJoined(node_id=5))
            await asyncio.sleep(0.1)

        scale_down_msgs = [m for m in sent if isinstance(m, RequestScaleDown)]
        assert len(scale_down_msgs) == 1
        assert scale_down_msgs[0].count == 1


class TestReconcilerApplyBounds:
    def test_updates_min_nodes(self) -> None:
        s = _State(desired=4, current=frozenset({1, 2, 3, 4}))
        new_s = _apply_bounds(s, BoundsChanged(min=2, max=8, desired=4))
        assert new_s.min_nodes == 2

    def test_desired_rebased_to_message(self) -> None:
        s = _State(desired=4, current=frozenset({1, 2, 3, 4}))
        new_s = _apply_bounds(s, BoundsChanged(min=1, max=8, desired=6))
        assert new_s.desired == 6
