"""Tests for SessionProjection — mutable projection that builds SessionView from domain events."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from skyward.api.events import (
    Error,
    Log,
    Metric,
    Node,
    Pool,
    Scaling,
    Task,
)
from skyward.api.views import (
    NodeStatus,
    PoolPhase,
    SessionView,
    TasksView,
)

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


def _make_projection(**kwargs):
    from skyward.api.projection import SessionProjection

    return SessionProjection(**kwargs)


def _provision(proj, name: str = "pool-1", total_nodes: int = 2) -> None:
    proj.handle(Pool.Provisioning(
        pool_name=name, total_nodes=total_nodes, started_at=1000.0,
    ))


class TestPoolLifecycle:
    def test_provisioning_creates_pool(self) -> None:
        proj = _make_projection()
        _provision(proj, "my-pool", total_nodes=4)

        view = proj.view
        assert "my-pool" in view.pools
        pv = view.pools["my-pool"]
        assert pv.phase == PoolPhase.PROVISIONING
        assert pv.total_nodes == 4
        assert pv.started_at == 1000.0

    def test_phase_changed_advances_phase(self) -> None:
        proj = _make_projection()
        _provision(proj)
        proj.handle(Pool.PhaseChanged(pool_name="pool-1", phase="SSH"))

        assert proj.view.pools["pool-1"].phase == PoolPhase.SSH

    def test_phase_changed_only_advances_forward(self) -> None:
        proj = _make_projection()
        _provision(proj)
        proj.handle(Pool.PhaseChanged(pool_name="pool-1", phase="READY"))
        proj.handle(Pool.PhaseChanged(pool_name="pool-1", phase="SSH"))

        assert proj.view.pools["pool-1"].phase == PoolPhase.READY

    def test_stopped_sets_stopping(self) -> None:
        proj = _make_projection()
        _provision(proj)
        proj.handle(Pool.Stopped(pool_name="pool-1"))

        assert proj.view.pools["pool-1"].phase == PoolPhase.STOPPING

    def test_provision_failed_no_state_change(self) -> None:
        proj = _make_projection()
        _provision(proj)
        proj.handle(Pool.ProvisionFailed(pool_name="pool-1", reason="quota"))

        assert proj.view.pools["pool-1"].phase == PoolPhase.PROVISIONING

    def test_unknown_pool_events_ignored(self) -> None:
        proj = _make_projection()
        proj.handle(Pool.PhaseChanged(pool_name="unknown", phase="READY"))
        assert "unknown" not in proj.view.pools

    def test_unknown_pool_stopped_ignored(self) -> None:
        proj = _make_projection()
        proj.handle(Pool.Stopped(pool_name="unknown"))
        assert len(proj.view.pools) == 0


class TestNodeLifecycle:
    def test_connected_sets_ssh_status(self) -> None:
        proj = _make_projection()
        _provision(proj, total_nodes=2)
        proj.handle(Node.Connected(pool_name="pool-1", node_id=0, instance=None))

        nv = proj.view.pools["pool-1"].nodes[0]
        assert nv.status == NodeStatus.SSH

    def test_connected_with_instance(self) -> None:
        proj = _make_projection()
        _provision(proj, total_nodes=1)
        mock_instance = MagicMock()
        proj.handle(Node.Connected(pool_name="pool-1", node_id=0, instance=mock_instance))

        nv = proj.view.pools["pool-1"].nodes[0]
        assert nv.instance is mock_instance

    def test_all_connected_advances_to_bootstrap(self) -> None:
        proj = _make_projection()
        _provision(proj, total_nodes=2)
        proj.handle(Node.Connected(pool_name="pool-1", node_id=0, instance=None))
        assert proj.view.pools["pool-1"].phase == PoolPhase.PROVISIONING

        proj.handle(Node.Connected(pool_name="pool-1", node_id=1, instance=None))
        assert proj.view.pools["pool-1"].phase == PoolPhase.BOOTSTRAP

    def test_ready_sets_node_ready(self) -> None:
        proj = _make_projection()
        _provision(proj, total_nodes=1)
        proj.handle(Node.Connected(pool_name="pool-1", node_id=0, instance=None))
        proj.handle(Node.Ready(pool_name="pool-1", node_id=0))

        nv = proj.view.pools["pool-1"].nodes[0]
        assert nv.status == NodeStatus.READY

    def test_all_ready_advances_to_ready_and_sets_ready_at(self) -> None:
        proj = _make_projection()
        _provision(proj, total_nodes=2)
        proj.handle(Node.Connected(pool_name="pool-1", node_id=0, instance=None))
        proj.handle(Node.Connected(pool_name="pool-1", node_id=1, instance=None))
        proj.handle(Node.Ready(pool_name="pool-1", node_id=0))
        assert proj.view.pools["pool-1"].phase != PoolPhase.READY

        proj.handle(Node.Ready(pool_name="pool-1", node_id=1))
        pv = proj.view.pools["pool-1"]
        assert pv.phase == PoolPhase.READY
        assert pv.ready_at > 0

    def test_ready_clears_bootstrap(self) -> None:
        proj = _make_projection()
        _provision(proj, total_nodes=1)
        proj.handle(Node.Connected(pool_name="pool-1", node_id=0, instance=None))
        proj.handle(Node.Bootstrap.Started(pool_name="pool-1", node_id=0, phase="apt"))
        assert proj.view.pools["pool-1"].nodes[0].bootstrap is not None

        proj.handle(Node.Ready(pool_name="pool-1", node_id=0))
        assert proj.view.pools["pool-1"].nodes[0].bootstrap is None

    def test_lost_removes_node(self) -> None:
        proj = _make_projection()
        _provision(proj, total_nodes=2)
        proj.handle(Node.Connected(pool_name="pool-1", node_id=0, instance=None))
        proj.handle(Node.Connected(pool_name="pool-1", node_id=1, instance=None))

        proj.handle(Node.Lost(pool_name="pool-1", node_id=0, reason="timeout"))
        assert 0 not in proj.view.pools["pool-1"].nodes
        assert 1 in proj.view.pools["pool-1"].nodes

    def test_node_event_for_unknown_pool_ignored(self) -> None:
        proj = _make_projection()
        proj.handle(Node.Connected(pool_name="unknown", node_id=0, instance=None))
        assert len(proj.view.pools) == 0


class TestBootstrapTimeline:
    def test_started_creates_bootstrap(self) -> None:
        proj = _make_projection()
        _provision(proj, total_nodes=1)
        proj.handle(Node.Connected(pool_name="pool-1", node_id=0, instance=None))
        proj.handle(Node.Bootstrap.Started(pool_name="pool-1", node_id=0, phase="apt"))

        bv = proj.view.pools["pool-1"].nodes[0].bootstrap
        assert bv is not None
        assert bv.active == "apt"
        assert "apt" in bv.phases

    def test_started_inserts_phase_respecting_order(self) -> None:
        proj = _make_projection()
        _provision(proj, total_nodes=1)
        proj.handle(Node.Bootstrap.Started(pool_name="pool-1", node_id=0, phase="apt"))
        proj.handle(Node.Bootstrap.Started(pool_name="pool-1", node_id=0, phase="uv"))
        proj.handle(Node.Bootstrap.Started(pool_name="pool-1", node_id=0, phase="env"))

        bv = proj.view.pools["pool-1"].nodes[0].bootstrap
        assert bv is not None
        phases = bv.phases
        assert phases.index("env") < phases.index("apt")
        assert phases.index("apt") < phases.index("uv")

    def test_completed_adds_to_set(self) -> None:
        proj = _make_projection()
        _provision(proj, total_nodes=1)
        proj.handle(Node.Bootstrap.Started(pool_name="pool-1", node_id=0, phase="apt"))
        proj.handle(Node.Bootstrap.Completed(pool_name="pool-1", node_id=0, phase="apt"))

        bv = proj.view.pools["pool-1"].nodes[0].bootstrap
        assert bv is not None
        assert "apt" in bv.completed

    def test_output_updates_text(self) -> None:
        proj = _make_projection()
        _provision(proj, total_nodes=1)
        proj.handle(Node.Bootstrap.Started(pool_name="pool-1", node_id=0, phase="apt"))
        proj.handle(Node.Bootstrap.Output(pool_name="pool-1", node_id=0, output="installing..."))

        bv = proj.view.pools["pool-1"].nodes[0].bootstrap
        assert bv is not None
        assert bv.output == "installing..."

    def test_done_success_sets_bootstrapping(self) -> None:
        proj = _make_projection()
        _provision(proj, total_nodes=1)
        proj.handle(Node.Connected(pool_name="pool-1", node_id=0, instance=None))
        proj.handle(Node.Bootstrap.Done(pool_name="pool-1", node_id=0, success=True))

        nv = proj.view.pools["pool-1"].nodes[0]
        assert nv.status == NodeStatus.BOOTSTRAPPING

    def test_all_bootstrap_done_advances_to_workers(self) -> None:
        proj = _make_projection()
        _provision(proj, total_nodes=2)
        proj.handle(Node.Connected(pool_name="pool-1", node_id=0, instance=None))
        proj.handle(Node.Connected(pool_name="pool-1", node_id=1, instance=None))

        proj.handle(Node.Bootstrap.Done(pool_name="pool-1", node_id=0, success=True))
        assert proj.view.pools["pool-1"].phase != PoolPhase.WORKERS

        proj.handle(Node.Bootstrap.Done(pool_name="pool-1", node_id=1, success=True))
        assert proj.view.pools["pool-1"].phase == PoolPhase.WORKERS

    def test_done_failure_clears_bootstrap(self) -> None:
        proj = _make_projection()
        _provision(proj, total_nodes=1)
        proj.handle(Node.Bootstrap.Started(pool_name="pool-1", node_id=0, phase="apt"))
        assert proj.view.pools["pool-1"].nodes[0].bootstrap is not None

        proj.handle(Node.Bootstrap.Done(pool_name="pool-1", node_id=0, success=False))
        assert proj.view.pools["pool-1"].nodes[0].bootstrap is None

    def test_command_updates_output(self) -> None:
        proj = _make_projection()
        _provision(proj, total_nodes=1)
        proj.handle(Node.Bootstrap.Started(pool_name="pool-1", node_id=0, phase="apt"))
        long_cmd = "x" * 200
        proj.handle(Node.Bootstrap.Command(pool_name="pool-1", node_id=0, command=long_cmd))

        bv = proj.view.pools["pool-1"].nodes[0].bootstrap
        assert bv is not None
        assert len(bv.output) <= 80

    def test_output_on_missing_bootstrap_ignored(self) -> None:
        proj = _make_projection()
        _provision(proj, total_nodes=1)
        proj.handle(Node.Connected(pool_name="pool-1", node_id=0, instance=None))
        proj.handle(Node.Bootstrap.Output(pool_name="pool-1", node_id=0, output="text"))
        nv = proj.view.pools["pool-1"].nodes[0]
        assert nv.bootstrap is None


class TestTasks:
    def test_queued_creates_entry(self) -> None:
        proj = _make_projection()
        _provision(proj, total_nodes=1)
        proj.handle(Task.Queued(
            pool_name="pool-1", task_id="t1", name="train(x)", kind="single",
        ))

        pv = proj.view.pools["pool-1"]
        assert pv.tasks.queued == 1
        assert "t1" in pv.tasks.inflight
        entry = pv.tasks.inflight["t1"]
        assert entry.name == "train(x)"
        assert entry.kind == "single"

    def test_queued_sets_first_task_at(self) -> None:
        proj = _make_projection()
        _provision(proj, total_nodes=1)
        proj.handle(Task.Queued(
            pool_name="pool-1", task_id="t1", name="f", kind="single",
        ))
        first = proj.view.pools["pool-1"].tasks.first_task_at
        assert first > 0

        proj.handle(Task.Queued(
            pool_name="pool-1", task_id="t2", name="g", kind="single",
        ))
        assert proj.view.pools["pool-1"].tasks.first_task_at == first

    def test_assigned_updates_entry(self) -> None:
        proj = _make_projection()
        _provision(proj, total_nodes=1)
        proj.handle(Task.Queued(
            pool_name="pool-1", task_id="t1", name="f", kind="single",
        ))
        proj.handle(Task.Assigned(pool_name="pool-1", task_id="t1", node_id=0))

        pv = proj.view.pools["pool-1"]
        assert pv.tasks.queued == 0
        assert pv.tasks.running == 1
        assert pv.tasks.inflight["t1"].node_id == 0

    def test_assigned_already_assigned_no_double_count(self) -> None:
        proj = _make_projection()
        _provision(proj, total_nodes=1)
        proj.handle(Task.Queued(
            pool_name="pool-1", task_id="t1", name="f", kind="single",
        ))
        proj.handle(Task.Assigned(pool_name="pool-1", task_id="t1", node_id=0))
        proj.handle(Task.Assigned(pool_name="pool-1", task_id="t1", node_id=1))

        pv = proj.view.pools["pool-1"]
        assert pv.tasks.queued == 0
        assert pv.tasks.running == 1

    def test_completed_records_latency_and_fn_stats(self) -> None:
        proj = _make_projection()
        _provision(proj, total_nodes=1)
        proj.handle(Task.Queued(
            pool_name="pool-1", task_id="t1", name="train(x)", kind="single",
        ))
        proj.handle(Task.Assigned(pool_name="pool-1", task_id="t1", node_id=0))
        proj.handle(Task.Completed(
            pool_name="pool-1", task_id="t1", node_id=0, elapsed=2.5,
        ))

        pv = proj.view.pools["pool-1"]
        assert pv.tasks.running == 0
        assert pv.tasks.done == 1
        assert 2.5 in pv.tasks.latencies
        assert "train" in pv.tasks.fn_stats
        assert 2.5 in pv.tasks.fn_stats["train"]
        assert pv.tasks.tasks_per_node[0] == 1
        assert "t1" not in pv.tasks.inflight

    def test_failed_increments_failed(self) -> None:
        proj = _make_projection()
        _provision(proj, total_nodes=1)
        proj.handle(Task.Queued(
            pool_name="pool-1", task_id="t1", name="train(x)", kind="single",
        ))
        proj.handle(Task.Assigned(pool_name="pool-1", task_id="t1", node_id=0))
        proj.handle(Task.Failed(
            pool_name="pool-1", task_id="t1", node_id=0, error="ValueError",
        ))

        pv = proj.view.pools["pool-1"]
        assert pv.tasks.running == 0
        assert pv.tasks.failed == 1
        assert "train" in pv.tasks.fn_failed
        assert pv.tasks.fn_failed["train"] == 1
        assert "t1" not in pv.tasks.inflight

    def test_broadcast_partial_increments(self) -> None:
        proj = _make_projection()
        _provision(proj, total_nodes=2)
        proj.handle(Task.Queued(
            pool_name="pool-1", task_id="t1", name="eval", kind="broadcast",
            broadcast_total=2,
        ))
        proj.handle(Task.BroadcastPartial(pool_name="pool-1", task_id="t1"))

        entry = proj.view.pools["pool-1"].tasks.inflight["t1"]
        assert entry.broadcast_done == 1

    def test_task_event_for_unknown_pool_ignored(self) -> None:
        proj = _make_projection()
        proj.handle(Task.Queued(
            pool_name="unknown", task_id="t1", name="f", kind="single",
        ))
        assert len(proj.view.pools) == 0


class TestMetrics:
    def test_sampled_updates_node_metrics(self) -> None:
        proj = _make_projection()
        _provision(proj, total_nodes=1)
        proj.handle(Node.Connected(pool_name="pool-1", node_id=0, instance=None))
        proj.handle(Metric.Sampled(
            pool_name="pool-1", node_id=0, name="gpu_util", value=0.95,
        ))

        nv = proj.view.pools["pool-1"].nodes[0]
        assert nv.metrics["gpu_util"] == 0.95

    def test_sampled_multiple_metrics(self) -> None:
        proj = _make_projection()
        _provision(proj, total_nodes=1)
        proj.handle(Node.Connected(pool_name="pool-1", node_id=0, instance=None))
        proj.handle(Metric.Sampled(
            pool_name="pool-1", node_id=0, name="gpu_util", value=0.5,
        ))
        proj.handle(Metric.Sampled(
            pool_name="pool-1", node_id=0, name="mem_used", value=32.0,
        ))

        nv = proj.view.pools["pool-1"].nodes[0]
        assert nv.metrics["gpu_util"] == 0.5
        assert nv.metrics["mem_used"] == 32.0


class TestScaling:
    def test_desired_changed_scaling_up(self) -> None:
        proj = _make_projection()
        _provision(proj, total_nodes=2)
        proj.handle(Scaling.DesiredChanged(
            pool_name="pool-1", desired=4, reason="pressure",
        ))

        sv = proj.view.pools["pool-1"].scaling
        assert sv.desired == 4
        assert sv.reconciler_state == "scaling_up"

    def test_desired_changed_draining(self) -> None:
        proj = _make_projection()
        _provision(proj, total_nodes=2)
        proj.handle(Node.Connected(pool_name="pool-1", node_id=0, instance=None))
        proj.handle(Node.Connected(pool_name="pool-1", node_id=1, instance=None))
        proj.handle(Scaling.DesiredChanged(
            pool_name="pool-1", desired=2, reason="initial",
        ))
        proj.handle(Scaling.DesiredChanged(
            pool_name="pool-1", desired=1, reason="low load",
        ))

        sv = proj.view.pools["pool-1"].scaling
        assert sv.desired == 1
        assert sv.reconciler_state == "draining"

    def test_desired_changed_watching(self) -> None:
        proj = _make_projection()
        _provision(proj, total_nodes=2)
        proj.handle(Node.Connected(pool_name="pool-1", node_id=0, instance=None))
        proj.handle(Node.Connected(pool_name="pool-1", node_id=1, instance=None))
        proj.handle(Scaling.DesiredChanged(
            pool_name="pool-1", desired=4, reason="initial",
        ))
        proj.handle(Scaling.DesiredChanged(
            pool_name="pool-1", desired=2, reason="stable",
        ))

        sv = proj.view.pools["pool-1"].scaling
        assert sv.reconciler_state == "watching"

    def test_spawning_increments(self) -> None:
        proj = _make_projection()
        _provision(proj, total_nodes=2)
        proj.handle(Scaling.Spawning(
            pool_name="pool-1", count=2, instances=("i-1", "i-2"),
        ))

        pv = proj.view.pools["pool-1"]
        assert pv.scaling.pending == 2
        assert pv.total_nodes == 4

    def test_draining_increments(self) -> None:
        proj = _make_projection()
        _provision(proj, total_nodes=2)
        proj.handle(Scaling.Draining(pool_name="pool-1", node_id=0))

        sv = proj.view.pools["pool-1"].scaling
        assert sv.draining == 1
        assert sv.reconciler_state == "draining"

    def test_drain_completed(self) -> None:
        proj = _make_projection()
        _provision(proj, total_nodes=2)
        proj.handle(Node.Connected(pool_name="pool-1", node_id=0, instance=None))
        proj.handle(Scaling.Draining(pool_name="pool-1", node_id=0))
        proj.handle(Scaling.DrainCompleted(pool_name="pool-1", node_id=0))

        pv = proj.view.pools["pool-1"]
        assert pv.scaling.draining == 0
        assert 0 not in pv.nodes
        assert pv.total_nodes == 1


class TestCallbacks:
    def test_on_change_fires_on_state_change(self) -> None:
        changes: list[tuple[SessionView, SessionView]] = []

        def on_change(old: SessionView, new: SessionView) -> None:
            changes.append((old, new))

        proj = _make_projection(on_change=on_change)
        _provision(proj)

        assert len(changes) == 1
        old, new = changes[0]
        assert len(old.pools) == 0
        assert "pool-1" in new.pools

    def test_on_log_fires_on_log_emitted(self) -> None:
        logs: list[Log.Emitted] = []

        def on_log(event: Log.Emitted) -> None:
            logs.append(event)

        proj = _make_projection(on_log=on_log)
        _provision(proj)
        proj.handle(Log.Emitted(pool_name="pool-1", node_id=0, message="hello"))

        assert len(logs) == 1
        assert logs[0].message == "hello"

    def test_log_does_not_fire_on_change(self) -> None:
        changes: list[tuple[SessionView, SessionView]] = []

        def on_change(old: SessionView, new: SessionView) -> None:
            changes.append((old, new))

        proj = _make_projection(on_change=on_change)
        _provision(proj)
        change_count_after_provision = len(changes)

        proj.handle(Log.Emitted(pool_name="pool-1", node_id=0, message="hello"))
        assert len(changes) == change_count_after_provision

    def test_on_change_not_fired_when_state_unchanged(self) -> None:
        changes: list[tuple[SessionView, SessionView]] = []

        def on_change(old: SessionView, new: SessionView) -> None:
            changes.append((old, new))

        proj = _make_projection(on_change=on_change)
        _provision(proj)
        count = len(changes)

        proj.handle(Pool.ProvisionFailed(pool_name="pool-1", reason="quota"))
        assert len(changes) == count


class TestErrorAndUnknownEvents:
    def test_error_occurred_no_state_change(self) -> None:
        proj = _make_projection()
        _provision(proj)
        proj.handle(Error.Occurred(pool_name="pool-1", message="oops"))
        assert proj.view.pools["pool-1"].phase == PoolPhase.PROVISIONING

    def test_unknown_event_ignored(self) -> None:
        proj = _make_projection()
        _provision(proj)
        proj.handle("totally not an event")  # type: ignore[arg-type]
        assert proj.view.pools["pool-1"].phase == PoolPhase.PROVISIONING


class TestHelpers:
    def test_insert_phase_known_order(self) -> None:
        from skyward.api.projection import _insert_phase

        phases = ("connecting", "apt", "uv", "deps", "worker")
        result = _insert_phase(phases, "env")
        assert result.index("env") < result.index("apt")

    def test_insert_phase_already_present(self) -> None:
        from skyward.api.projection import _insert_phase

        phases = ("connecting", "apt", "worker")
        result = _insert_phase(phases, "apt")
        assert result == phases

    def test_insert_phase_unknown_before_worker(self) -> None:
        from skyward.api.projection import _insert_phase

        phases = ("connecting", "apt", "worker")
        result = _insert_phase(phases, "custom")
        assert result.index("custom") < result.index("worker")

    def test_advance_only_forward(self) -> None:
        from skyward.api.projection import _advance

        assert _advance(PoolPhase.PROVISIONING, PoolPhase.SSH) == PoolPhase.SSH
        assert _advance(PoolPhase.READY, PoolPhase.SSH) == PoolPhase.READY

    def test_throughput_calculation(self) -> None:
        from skyward.api.projection import _throughput

        tasks = TasksView(done=60, first_task_at=1000.0)
        result = _throughput(tasks, now=1060.0)
        assert result == pytest.approx(60.0, rel=0.01)

    def test_throughput_zero_when_no_tasks(self) -> None:
        from skyward.api.projection import _throughput

        tasks = TasksView()
        assert _throughput(tasks) == 0.0


class TestReconciled:
    def test_reconciled_syncs_from_snapshot(self) -> None:
        from skyward.actors.snapshot import (
            NodeSnapshot,
            PoolSnapshot,
            ScalingSnapshot,
            TaskCounters,
        )
        from skyward.actors.snapshot import NodeStatus as SnapshotNodeStatus
        from skyward.actors.snapshot import PoolPhase as SnapshotPhase

        proj = _make_projection()
        _provision(proj, total_nodes=2)

        snapshot = PoolSnapshot(
            name="pool-1",
            phase=SnapshotPhase.READY,
            nodes=(
                NodeSnapshot(node_id=0, instance_id="i-0", status=SnapshotNodeStatus.READY),
                NodeSnapshot(node_id=1, instance_id="i-1", status=SnapshotNodeStatus.SSH),
            ),
            tasks=TaskCounters(queued=1, running=2, done=10, failed=1),
            scaling=ScalingSnapshot(
                desired_nodes=4, pending_nodes=1, draining_nodes=0,
                reconciler_state="scaling_up", is_elastic=True,
                min_nodes=2, max_nodes=8,
            ),
            started_at=500.0,
        )
        proj.handle(Pool.Reconciled(pool_name="pool-1", snapshot=snapshot))

        pv = proj.view.pools["pool-1"]
        assert pv.phase == PoolPhase.READY
        assert pv.total_nodes == 2
        assert pv.started_at == 500.0
        assert len(pv.nodes) == 2
        assert pv.nodes[0].status == NodeStatus.READY
        assert pv.nodes[1].status == NodeStatus.SSH
        assert pv.tasks.queued == 1
        assert pv.tasks.running == 2
        assert pv.tasks.done == 10
        assert pv.tasks.failed == 1
        assert pv.scaling.desired == 4
        assert pv.scaling.pending == 1
        assert pv.scaling.is_elastic is True
        assert pv.scaling.min_nodes == 2
        assert pv.scaling.max_nodes == 8

    def test_reconciled_none_snapshot_ignored(self) -> None:
        proj = _make_projection()
        _provision(proj)
        before_phase = proj.view.pools["pool-1"].phase
        proj.handle(Pool.Reconciled(pool_name="pool-1", snapshot=None))
        assert proj.view.pools["pool-1"].phase == before_phase


class TestSettableCallbacks:
    def test_on_change_settable_after_init(self) -> None:
        proj = _make_projection()
        changes: list[tuple[SessionView, SessionView]] = []

        proj.on_change = lambda old, new: changes.append((old, new))
        _provision(proj)

        assert len(changes) == 1
        old, new = changes[0]
        assert len(old.pools) == 0
        assert "pool-1" in new.pools

    def test_on_log_settable_after_init(self) -> None:
        proj = _make_projection()
        logs: list[Log.Emitted] = []

        proj.on_log = lambda event: logs.append(event)
        _provision(proj)
        proj.handle(Log.Emitted(pool_name="pool-1", node_id=0, message="hello"))

        assert len(logs) == 1
        assert logs[0].message == "hello"

    def test_on_event_fires_for_every_event(self) -> None:
        from skyward.api.events import SessionEvent

        events: list[SessionEvent] = []
        proj = _make_projection(on_event=lambda e: events.append(e))

        _provision(proj)
        proj.handle(Node.Connected(pool_name="pool-1", node_id=0, instance=None))
        proj.handle(Log.Emitted(pool_name="pool-1", node_id=0, message="hi"))

        assert len(events) == 3
        assert isinstance(events[0], Pool.Provisioning)
        assert isinstance(events[1], Node.Connected)
        assert isinstance(events[2], Log.Emitted)

    def test_on_event_fires_for_log_events_too(self) -> None:
        from skyward.api.events import SessionEvent

        events: list[SessionEvent] = []
        logs: list[Log.Emitted] = []

        proj = _make_projection(
            on_event=lambda e: events.append(e),
            on_log=lambda e: logs.append(e),
        )
        proj.handle(Log.Emitted(pool_name="pool-1", node_id=0, message="msg"))

        assert len(events) == 1
        assert isinstance(events[0], Log.Emitted)
        assert len(logs) == 1
        assert logs[0].message == "msg"
