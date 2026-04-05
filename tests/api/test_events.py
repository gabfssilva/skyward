from __future__ import annotations

import cloudpickle
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

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


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
