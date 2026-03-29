from __future__ import annotations

import time

import cloudpickle
import pytest

from skyward.api.events import (
    Error,
    Log,
    Metric,
    Node,
    Pool,
    Scaling,
    SessionEvent,
    Task,
)

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class TestPoolEvents:
    def test_provisioning(self):
        now = time.time()
        e = Pool.Provisioning(pool_name="p1", total_nodes=4, started_at=now)
        assert e.pool_name == "p1"
        assert e.total_nodes == 4
        assert e.started_at == now

    def test_phase_changed(self):
        e = Pool.PhaseChanged(pool_name="p1", phase="ready")
        assert e.phase == "ready"

    def test_stopped(self):
        e = Pool.Stopped(pool_name="p1")
        assert e.pool_name == "p1"

    def test_reconciled(self):
        e = Pool.Reconciled(pool_name="p1", snapshot={"mock": True})
        assert e.snapshot == {"mock": True}

    def test_provision_failed(self):
        e = Pool.ProvisionFailed(pool_name="p1", reason="quota exceeded")
        assert e.reason == "quota exceeded"

    def test_frozen(self):
        e = Pool.Provisioning(pool_name="p1", total_nodes=4, started_at=0.0)
        with pytest.raises(AttributeError):
            e.pool_name = "other"  # type: ignore[misc]


class TestNodeEvents:
    def test_connected(self):
        e = Node.Connected(pool_name="p1", node_id=0, instance=None)
        assert e.node_id == 0
        assert e.instance is None

    def test_ready(self):
        e = Node.Ready(pool_name="p1", node_id=1)
        assert e.node_id == 1

    def test_lost(self):
        e = Node.Lost(pool_name="p1", node_id=2, reason="timeout")
        assert e.reason == "timeout"

    def test_connection_failed(self):
        e = Node.ConnectionFailed(pool_name="p1", error="refused")
        assert e.error == "refused"

    def test_preempted(self):
        e = Node.Preempted(pool_name="p1", reason="spot reclaimed")
        assert e.reason == "spot reclaimed"

    def test_worker_failed(self):
        e = Node.WorkerFailed(pool_name="p1", error="OOM")
        assert e.error == "OOM"

    def test_frozen(self):
        e = Node.Ready(pool_name="p1", node_id=0)
        with pytest.raises(AttributeError):
            e.node_id = 5  # type: ignore[misc]


class TestNodeBootstrapEvents:
    def test_started(self):
        e = Node.Bootstrap.Started(pool_name="p1", node_id=0, phase="apt")
        assert e.phase == "apt"

    def test_completed(self):
        e = Node.Bootstrap.Completed(pool_name="p1", node_id=0, phase="uv")
        assert e.phase == "uv"

    def test_output(self):
        e = Node.Bootstrap.Output(pool_name="p1", node_id=0, output="installing...")
        assert e.output == "installing..."

    def test_done_defaults(self):
        e = Node.Bootstrap.Done(pool_name="p1", node_id=0, success=True)
        assert e.error is None

    def test_done_with_error(self):
        e = Node.Bootstrap.Done(pool_name="p1", node_id=0, success=False, error="failed")
        assert e.error == "failed"

    def test_failed(self):
        e = Node.Bootstrap.Failed(pool_name="p1", node_id=0, phase="deps", error="timeout")
        assert e.error == "timeout"

    def test_command(self):
        e = Node.Bootstrap.Command(pool_name="p1", node_id=0, command="apt-get install")
        assert e.command == "apt-get install"

    def test_frozen(self):
        e = Node.Bootstrap.Started(pool_name="p1", node_id=0, phase="apt")
        with pytest.raises(AttributeError):
            e.phase = "other"  # type: ignore[misc]


class TestTaskEvents:
    def test_queued_defaults(self):
        e = Task.Queued(pool_name="p1", task_id="t1", name="train", kind="single")
        assert e.broadcast_total == 0

    def test_queued_broadcast(self):
        e = Task.Queued(pool_name="p1", task_id="t1", name="train", kind="broadcast", broadcast_total=4)
        assert e.broadcast_total == 4

    def test_assigned(self):
        e = Task.Assigned(pool_name="p1", task_id="t1", node_id=0)
        assert e.node_id == 0

    def test_completed(self):
        e = Task.Completed(pool_name="p1", task_id="t1", node_id=0, elapsed=1.5)
        assert e.elapsed == 1.5

    def test_failed(self):
        e = Task.Failed(pool_name="p1", task_id="t1", node_id=0, error="ValueError")
        assert e.error == "ValueError"

    def test_broadcast_partial(self):
        e = Task.BroadcastPartial(pool_name="p1", task_id="t1")
        assert e.task_id == "t1"

    def test_frozen(self):
        e = Task.Assigned(pool_name="p1", task_id="t1", node_id=0)
        with pytest.raises(AttributeError):
            e.task_id = "other"  # type: ignore[misc]


class TestMetricEvents:
    def test_sampled(self):
        e = Metric.Sampled(pool_name="p1", node_id=0, name="gpu_util", value=0.95)
        assert e.name == "gpu_util"
        assert e.value == 0.95

    def test_frozen(self):
        e = Metric.Sampled(pool_name="p1", node_id=0, name="gpu_util", value=0.5)
        with pytest.raises(AttributeError):
            e.value = 1.0  # type: ignore[misc]


class TestLogEvents:
    def test_emitted_defaults(self):
        e = Log.Emitted(pool_name="p1", node_id=0, message="hello")
        assert e.level == "info"

    def test_emitted_custom_level(self):
        e = Log.Emitted(pool_name="p1", node_id=0, message="bad", level="error")
        assert e.level == "error"

    def test_frozen(self):
        e = Log.Emitted(pool_name="p1", node_id=0, message="hello")
        with pytest.raises(AttributeError):
            e.message = "other"  # type: ignore[misc]


class TestScalingEvents:
    def test_desired_changed(self):
        e = Scaling.DesiredChanged(pool_name="p1", desired=6, reason="pressure")
        assert e.desired == 6

    def test_spawning_defaults(self):
        e = Scaling.Spawning(pool_name="p1", count=2)
        assert e.instances == ()

    def test_spawning_with_instances(self):
        e = Scaling.Spawning(pool_name="p1", count=2, instances=("i-1", "i-2"))
        assert len(e.instances) == 2

    def test_draining(self):
        e = Scaling.Draining(pool_name="p1", node_id=3)
        assert e.node_id == 3

    def test_drain_completed(self):
        e = Scaling.DrainCompleted(pool_name="p1", node_id=3)
        assert e.node_id == 3

    def test_frozen(self):
        e = Scaling.DesiredChanged(pool_name="p1", desired=6, reason="x")
        with pytest.raises(AttributeError):
            e.desired = 10  # type: ignore[misc]


class TestErrorEvents:
    def test_occurred_defaults(self):
        e = Error.Occurred(pool_name="p1", message="oops")
        assert e.fatal is False

    def test_occurred_fatal(self):
        e = Error.Occurred(pool_name="p1", message="crash", fatal=True)
        assert e.fatal is True

    def test_frozen(self):
        e = Error.Occurred(pool_name="p1", message="oops")
        with pytest.raises(AttributeError):
            e.fatal = True  # type: ignore[misc]


class TestCloudpickleRoundtrip:
    """Every event must survive cloudpickle serialization."""

    @pytest.fixture(
        params=[
            lambda: Pool.Provisioning(pool_name="p", total_nodes=1, started_at=0.0),
            lambda: Pool.PhaseChanged(pool_name="p", phase="ready"),
            lambda: Pool.Stopped(pool_name="p"),
            lambda: Pool.Reconciled(pool_name="p", snapshot=None),
            lambda: Pool.ProvisionFailed(pool_name="p", reason="err"),
            lambda: Node.Connected(pool_name="p", node_id=0, instance=None),
            lambda: Node.Ready(pool_name="p", node_id=0),
            lambda: Node.Lost(pool_name="p", node_id=0, reason="x"),
            lambda: Node.ConnectionFailed(pool_name="p", error="x"),
            lambda: Node.Preempted(pool_name="p", reason="x"),
            lambda: Node.WorkerFailed(pool_name="p", error="x"),
            lambda: Node.Bootstrap.Started(pool_name="p", node_id=0, phase="apt"),
            lambda: Node.Bootstrap.Completed(pool_name="p", node_id=0, phase="apt"),
            lambda: Node.Bootstrap.Output(pool_name="p", node_id=0, output="x"),
            lambda: Node.Bootstrap.Done(pool_name="p", node_id=0, success=True),
            lambda: Node.Bootstrap.Failed(pool_name="p", node_id=0, phase="apt", error="x"),
            lambda: Node.Bootstrap.Command(pool_name="p", node_id=0, command="echo"),
            lambda: Task.Queued(pool_name="p", task_id="t", name="f", kind="single"),
            lambda: Task.Assigned(pool_name="p", task_id="t", node_id=0),
            lambda: Task.Completed(pool_name="p", task_id="t", node_id=0, elapsed=1.0),
            lambda: Task.Failed(pool_name="p", task_id="t", node_id=0, error="x"),
            lambda: Task.BroadcastPartial(pool_name="p", task_id="t"),
            lambda: Metric.Sampled(pool_name="p", node_id=0, name="x", value=1.0),
            lambda: Log.Emitted(pool_name="p", node_id=0, message="x"),
            lambda: Scaling.DesiredChanged(pool_name="p", desired=4, reason="x"),
            lambda: Scaling.Spawning(pool_name="p", count=2),
            lambda: Scaling.Draining(pool_name="p", node_id=0),
            lambda: Scaling.DrainCompleted(pool_name="p", node_id=0),
            lambda: Error.Occurred(pool_name="p", message="x"),
        ],
    )
    def event(self, request):
        return request.param()

    def test_roundtrip(self, event):
        data = cloudpickle.dumps(event)
        restored = cloudpickle.loads(data)
        assert restored == event


class TestSessionEventTypeAlias:
    """SessionEvent union type alias must exist and cover all event types."""

    def test_alias_exists(self):
        assert SessionEvent is not None

    def test_pool_event_is_session_event(self):
        e = Pool.Provisioning(pool_name="p", total_nodes=1, started_at=0.0)
        assert isinstance(e, Pool.Provisioning)

    def test_node_bootstrap_is_session_event(self):
        e = Node.Bootstrap.Started(pool_name="p", node_id=0, phase="apt")
        assert isinstance(e, Node.Bootstrap.Started)

    def test_task_event_is_session_event(self):
        e = Task.Queued(pool_name="p", task_id="t", name="f", kind="single")
        assert isinstance(e, Task.Queued)

    def test_error_event_is_session_event(self):
        e = Error.Occurred(pool_name="p", message="x")
        assert isinstance(e, Error.Occurred)
