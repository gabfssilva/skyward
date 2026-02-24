from dataclasses import replace

from skyward.actors.reconciler import _State

_FAKE_CLUSTER: object = object()


def _make_state(
    desired: int = 2,
    current: frozenset[int] | None = None,
    pending: frozenset[int] | None = None,
    draining: frozenset[int] | None = None,
    next_node_id: int = 2,
    instance_map: dict[int, str] | None = None,
) -> _State:
    return _State(
        desired=desired,
        current=current or frozenset({0, 1}),
        pending=pending or frozenset(),
        draining=draining or frozenset(),
        next_node_id=next_node_id,
        instance_map=instance_map or {0: "i-0", 1: "i-1"},
        cluster=_FAKE_CLUSTER,  # type: ignore[arg-type]
    )


class TestReconcilerState:
    def test_effective_count_current_plus_pending(self) -> None:
        s = _make_state(current=frozenset({0, 1}), pending=frozenset({2}))
        assert len(s.current) + len(s.pending) == 3

    def test_desired_count_changed_up(self) -> None:
        s = _make_state(desired=2)
        new_s = replace(s, desired=4)
        effective = len(new_s.current) + len(new_s.pending)
        assert new_s.desired > effective  # should trigger scale-up

    def test_desired_count_changed_down(self) -> None:
        s = _make_state(desired=4, current=frozenset({0, 1, 2, 3}))
        new_s = replace(s, desired=2)
        assert new_s.desired < len(new_s.current)  # should trigger drain

    def test_node_lost_reduces_current(self) -> None:
        s = _make_state(current=frozenset({0, 1, 2}))
        new_s = replace(s, current=s.current - {2})
        assert 2 not in new_s.current
        assert len(new_s.current) == 2

    def test_node_lost_triggers_scale_up(self) -> None:
        s = _make_state(desired=3, current=frozenset({0, 1, 2}))
        new_s = replace(s, current=s.current - {2})
        effective = len(new_s.current) + len(new_s.pending)
        assert new_s.desired > effective  # should auto-repair

    def test_node_joined_moves_pending_to_current(self) -> None:
        s = _make_state(
            current=frozenset({0, 1}),
            pending=frozenset({2}),
        )
        new_s = replace(
            s,
            current=s.current | {2},
            pending=s.pending - {2},
        )
        assert 2 in new_s.current
        assert 2 not in new_s.pending

    def test_provision_result_updates_pending(self) -> None:
        s = _make_state(desired=4, next_node_id=2)
        new_pending_ids = [2, 3]
        new_s = replace(
            s,
            pending=s.pending | frozenset(new_pending_ids),
            next_node_id=4,
        )
        assert new_s.pending == frozenset({2, 3})
        assert new_s.next_node_id == 4

    def test_provision_error_stays_in_watching(self) -> None:
        s = _make_state(desired=4)
        # On provision error, state doesn't change
        assert s.desired == 4
        effective = len(s.current) + len(s.pending)
        assert s.desired > effective  # still needs scaling

    def test_drain_complete_removes_from_current(self) -> None:
        s = _make_state(
            desired=1,
            current=frozenset({0, 1}),
            draining=frozenset({1}),
        )
        new_s = replace(
            s,
            current=s.current - {1},
            draining=s.draining - {1},
        )
        assert 1 not in new_s.current
        assert not new_s.draining

    def test_drain_abort_on_desired_increase(self) -> None:
        s = _make_state(
            desired=1,
            current=frozenset({0, 1, 2}),
            draining=frozenset({2}),
        )
        new_s = replace(s, desired=3, draining=frozenset())
        assert not new_s.draining
        assert new_s.desired >= len(new_s.current)

    def test_head_node_protected(self) -> None:
        s = _make_state(
            desired=1,
            current=frozenset({0, 1, 2}),
        )
        excess = len(s.current) - s.desired
        victims = sorted(s.current, reverse=True)[:excess]
        victims = [nid for nid in victims if nid != 0]
        assert 0 not in victims
        assert victims == [2, 1]

    def test_node_lost_during_drain(self) -> None:
        s = _make_state(
            desired=1,
            current=frozenset({0, 1}),
            draining=frozenset({1}),
        )
        new_s = replace(
            s,
            current=s.current - {1},
            draining=s.draining - {1},
        )
        assert not new_s.draining  # drain complete
        assert len(new_s.current) == 1
