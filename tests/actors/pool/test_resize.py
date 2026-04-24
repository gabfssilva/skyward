from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from skyward.actors.messages import BoundsChanged, DesiredCountChanged
from skyward.actors.pool.actor import _apply_resize
from skyward.actors.pool.state import PoolState
from skyward.actors.snapshot import ScalingSnapshot
from skyward.api.spec import Nodes


@dataclass(frozen=True, slots=True)
class _FakeSpec:
    """Minimal stand-in for PoolSpec — only ``nodes`` is touched by the reducer."""

    nodes: Nodes


def _state(initial_nodes: Nodes, initial_scaling: ScalingSnapshot | None = None) -> PoolState:
    return PoolState(
        spec=_FakeSpec(nodes=initial_nodes),  # type: ignore[arg-type]
        provider=object(),
        reply_to=object(),  # type: ignore[arg-type]
        scaling=initial_scaling or ScalingSnapshot(
            desired_nodes=initial_nodes.desired,
            is_elastic=initial_nodes.auto_scaling,
            min_nodes=initial_nodes.min or initial_nodes.desired,
            max_nodes=initial_nodes.max or initial_nodes.desired,
        ),
    )


@pytest.mark.unit
class TestApplyResize:
    def test_elastic_shape_emits_bounds_and_desired(self) -> None:
        s = _state(Nodes(desired=2))
        new_s, bounds, desired = _apply_resize(s, Nodes(desired=2, max=6))
        assert bounds == BoundsChanged(min=2, max=6, desired=2)
        assert desired == DesiredCountChanged(desired=2, reason="resize")
        assert new_s.scaling.desired_nodes == 2
        assert new_s.scaling.is_elastic is True
        assert new_s.scaling.min_nodes == 2
        assert new_s.scaling.max_nodes == 6
        assert new_s.spec.nodes == Nodes(desired=2, max=6)

    def test_fixed_shape_is_not_elastic(self) -> None:
        s = _state(Nodes(desired=4))
        new_s, bounds, desired = _apply_resize(s, Nodes(desired=3))
        assert bounds == BoundsChanged(min=3, max=3, desired=3)
        assert desired == DesiredCountChanged(desired=3, reason="resize")
        assert new_s.scaling.is_elastic is False
        assert new_s.scaling.min_nodes == 3
        assert new_s.scaling.max_nodes == 3
        assert new_s.spec.nodes == Nodes(desired=3)

    def test_full_nodes_shape_with_explicit_min_max(self) -> None:
        s = _state(Nodes(desired=4))
        new_s, bounds, desired = _apply_resize(s, Nodes(desired=4, min=2, max=8))
        assert bounds == BoundsChanged(min=2, max=8, desired=4)
        assert desired == DesiredCountChanged(desired=4, reason="resize")
        assert new_s.scaling.desired_nodes == 4
        assert new_s.scaling.is_elastic is True
        assert new_s.scaling.min_nodes == 2
        assert new_s.scaling.max_nodes == 8
        assert new_s.spec.nodes == Nodes(desired=4, min=2, max=8)

    def test_preserves_unrelated_scaling_fields(self) -> None:
        initial = ScalingSnapshot(
            desired_nodes=2,
            pending_nodes=1,
            draining_nodes=3,
            reconciler_state="scaling_up",
            is_elastic=False,
            min_nodes=2,
            max_nodes=2,
        )
        s = _state(Nodes(desired=2), initial_scaling=initial)
        new_s, _bounds, _desired = _apply_resize(s, Nodes(desired=5, max=10))
        assert new_s.scaling.pending_nodes == 1
        assert new_s.scaling.draining_nodes == 3
        assert new_s.scaling.reconciler_state == "scaling_up"

    def test_returns_new_state_without_mutating_input(self) -> None:
        s = _state(Nodes(desired=2))
        new_s, _b, _d = _apply_resize(s, Nodes(desired=4, max=8))
        assert s.spec.nodes == Nodes(desired=2)
        assert s.scaling.desired_nodes == 2
        assert new_s is not s
        assert replace(new_s.scaling, desired_nodes=s.scaling.desired_nodes) != new_s.scaling
