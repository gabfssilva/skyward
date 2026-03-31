from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from skyward.actors.spy_adapter import _format_task, _node_id_from_path, translate
from skyward.api.events import (
    Error as ErrorEvent,
    Log as LogEvent,
    Metric as MetricEvent,
    Node,
    Pool,
    Scaling,
    Task,
)

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


def _spy(event: object, actor_path: str = "/system/pool") -> object:
    from casty import SpyEvent

    return SpyEvent(actor_path=actor_path, event=event, timestamp=0.0)


def _node_instance(node: int = 0) -> MagicMock:
    ni = MagicMock()
    ni.node = node
    ni.instance = MagicMock()
    return ni


# ── Node events ─────────────────────────────────────────────────


class TestNodeEvents:
    def test_connected_extracts_node_id_from_path(self):
        from skyward.actors.node.messages import _Connected

        ni = _node_instance(3)
        ev = _Connected(transport_ref=MagicMock(), local_port=22, instance=ni)
        result = translate(_spy(ev, "/system/pool/node-3"), "pool")

        assert isinstance(result, Node.Connected)
        assert result.node_id == 3
        assert result.instance is ni.instance

    def test_connected_without_instance(self):
        from skyward.actors.node.messages import _Connected

        ev = _Connected(transport_ref=MagicMock(), local_port=22, instance=None)
        result = translate(_spy(ev, "/system/pool/node-1"), "pool")

        assert isinstance(result, Node.Connected)
        assert result.node_id == 1
        assert result.instance is None

    def test_connected_no_node_in_path_returns_none(self):
        from skyward.actors.node.messages import _Connected

        ni = _node_instance(0)
        ev = _Connected(transport_ref=MagicMock(), local_port=22, instance=ni)
        result = translate(_spy(ev, "/system/pool"), "pool")

        assert result is None

    def test_worker_started_becomes_ready(self):
        from skyward.actors.node.messages import _WorkerStarted

        ev = _WorkerStarted(local_port=25520, private_ip="10.0.0.1")
        result = translate(_spy(ev, "/system/pool/node-5"), "pool")

        assert isinstance(result, Node.Ready)
        assert result.node_id == 5
        assert result.pool_name == "pool"

    def test_worker_started_no_node_returns_none(self):
        from skyward.actors.node.messages import _WorkerStarted

        ev = _WorkerStarted(local_port=25520, private_ip="10.0.0.1")
        result = translate(_spy(ev, "/system/pool"), "pool")

        assert result is None

    def test_worker_failed(self):
        from skyward.actors.node.messages import _WorkerFailed

        ev = _WorkerFailed(error="OOM killed")
        result = translate(_spy(ev), "pool")

        assert isinstance(result, Node.WorkerFailed)
        assert result.error == "OOM killed"

    def test_node_lost(self):
        from skyward.actors.messages import NodeLost

        ev = NodeLost(node_id=2, reason="timeout")
        result = translate(_spy(ev), "pool")

        assert isinstance(result, Node.Lost)
        assert result.node_id == 2
        assert result.reason == "timeout"

    def test_connection_failed(self):
        from skyward.actors.node.messages import _ConnectionFailed

        ev = _ConnectionFailed(error="connection refused")
        result = translate(_spy(ev), "pool")

        assert isinstance(result, Node.ConnectionFailed)
        assert result.error == "connection refused"

    def test_preempted(self):
        from skyward.actors.messages import Preempted

        ev = Preempted(reason="spot reclaimed")
        result = translate(_spy(ev), "pool")

        assert isinstance(result, Node.Preempted)
        assert result.reason == "spot reclaimed"


# ── Bootstrap events ────────────────────────────────────────────


class TestBootstrapEvents:
    def test_phase_started(self):
        from skyward.actors.messages import BootstrapPhase

        ni = _node_instance(1)
        ev = BootstrapPhase(instance=ni, event="started", phase="apt")
        result = translate(_spy(ev), "pool")

        assert isinstance(result, Node.Bootstrap.Started)
        assert result.node_id == 1
        assert result.phase == "apt"

    def test_phase_started_bootstrap_phase_skipped(self):
        from skyward.actors.messages import BootstrapPhase

        ni = _node_instance(1)
        ev = BootstrapPhase(instance=ni, event="started", phase="bootstrap")
        result = translate(_spy(ev), "pool")

        assert result is None

    def test_phase_completed(self):
        from skyward.actors.messages import BootstrapPhase

        ni = _node_instance(2)
        ev = BootstrapPhase(instance=ni, event="completed", phase="uv")
        result = translate(_spy(ev), "pool")

        assert isinstance(result, Node.Bootstrap.Completed)
        assert result.node_id == 2
        assert result.phase == "uv"

    def test_phase_completed_bootstrap_phase_skipped(self):
        from skyward.actors.messages import BootstrapPhase

        ni = _node_instance(2)
        ev = BootstrapPhase(instance=ni, event="completed", phase="bootstrap")
        result = translate(_spy(ev), "pool")

        assert result is None

    def test_phase_failed(self):
        from skyward.actors.messages import BootstrapPhase

        ni = _node_instance(0)
        ev = BootstrapPhase(instance=ni, event="failed", phase="deps", error="timeout")
        result = translate(_spy(ev), "pool")

        assert isinstance(result, Node.Bootstrap.Failed)
        assert result.node_id == 0
        assert result.phase == "deps"
        assert result.error == "timeout"

    def test_phase_failed_no_error(self):
        from skyward.actors.messages import BootstrapPhase

        ni = _node_instance(0)
        ev = BootstrapPhase(instance=ni, event="failed", phase="deps")
        result = translate(_spy(ev), "pool")

        assert isinstance(result, Node.Bootstrap.Failed)
        assert result.error == ""

    def test_bootstrap_command(self):
        from skyward.actors.messages import BootstrapCommand

        ni = _node_instance(1)
        ev = BootstrapCommand(instance=ni, command="apt-get install python3")
        result = translate(_spy(ev), "pool")

        assert isinstance(result, Node.Bootstrap.Command)
        assert result.node_id == 1
        assert result.command == "apt-get install python3"

    def test_bootstrap_done(self):
        from skyward.actors.messages import BootstrapDone

        ni = _node_instance(3)
        ev = BootstrapDone(instance=ni, success=True, error=None)
        result = translate(_spy(ev), "pool")

        assert isinstance(result, Node.Bootstrap.Done)
        assert result.node_id == 3
        assert result.success is True
        assert result.error is None

    def test_bootstrap_done_failed(self):
        from skyward.actors.messages import BootstrapDone

        ni = _node_instance(3)
        ev = BootstrapDone(instance=ni, success=False, error="script exited 1")
        result = translate(_spy(ev), "pool")

        assert isinstance(result, Node.Bootstrap.Done)
        assert result.success is False
        assert result.error == "script exited 1"

    def test_bootstrap_console_output(self):
        from skyward.actors.messages import ConsoleOutput

        ni = _node_instance(2)
        ev = ConsoleOutput(instance=ni, content="  Installing deps  ")
        result = translate(_spy(ev), "pool")

        assert isinstance(result, Node.Bootstrap.Output)
        assert result.output == "Installing deps"

    def test_bootstrap_console_empty_skipped(self):
        from skyward.actors.messages import ConsoleOutput

        ni = _node_instance(2)
        ev = ConsoleOutput(instance=ni, content="   ")
        result = translate(_spy(ev), "pool")

        assert result is None

    def test_bootstrap_console_comment_skipped(self):
        from skyward.actors.messages import ConsoleOutput

        ni = _node_instance(2)
        ev = ConsoleOutput(instance=ni, content="# this is a comment")
        result = translate(_spy(ev), "pool")

        assert result is None

    def test_post_bootstrap_failed(self):
        from skyward.actors.node.messages import _PostBootstrapFailed

        ev = _PostBootstrapFailed(error="hook crashed")
        result = translate(_spy(ev), "pool")

        assert isinstance(result, ErrorEvent.Occurred)
        assert "Post-bootstrap failed" in result.message
        assert "hook crashed" in result.message


# ── Task events ─────────────────────────────────────────────────


class TestTaskEvents:
    def test_submit_task_queued(self):
        from skyward.actors.messages import SubmitTask

        def my_train(): ...

        ev = SubmitTask(fn=my_train, args=(1, 2), kwargs={"lr": 0.01}, reply_to=MagicMock(), task_id="t1")
        result = translate(_spy(ev), "pool")

        assert isinstance(result, Task.Queued)
        assert result.task_id == "t1"
        assert result.kind == "single"
        assert "my_train" in result.name

    def test_submit_broadcast_queued(self):
        from skyward.actors.messages import SubmitBroadcast

        def my_train(): ...

        ev = SubmitBroadcast(fn=my_train, args=(), kwargs={}, reply_to=MagicMock(), task_id="t2")
        result = translate(_spy(ev), "pool")

        assert isinstance(result, Task.Queued)
        assert result.task_id == "t2"
        assert result.kind == "broadcast"

    def test_task_submitted_assigned(self):
        from skyward.actors.messages import TaskSubmitted

        ev = TaskSubmitted(task_id="t1", node_id=3)
        result = translate(_spy(ev), "pool")

        assert isinstance(result, Task.Assigned)
        assert result.task_id == "t1"
        assert result.node_id == 3

    def test_task_result_success(self):
        from skyward.actors.messages import TaskSucceeded

        ev = TaskSucceeded(value="ok", node_id=1, task_id="t1")
        result = translate(_spy(ev), "pool")

        assert isinstance(result, Task.Completed)
        assert result.task_id == "t1"
        assert result.node_id == 1
        assert result.elapsed == 0.0

    def test_task_result_failure(self):
        from skyward.actors.messages import TaskFailed

        ev = TaskFailed(error=ValueError("x"), node_id=1, task_id="t1")
        result = translate(_spy(ev), "pool")

        assert isinstance(result, Task.Failed)
        assert result.task_id == "t1"
        assert result.node_id == 1

    def test_task_result_interrupted(self):
        from skyward.actors.messages import TaskInterrupted

        ev = TaskInterrupted(error=RuntimeError("lost"), node_id=1, task_id="t1")
        result = translate(_spy(ev), "pool")

        assert isinstance(result, Task.Failed)
        assert result.task_id == "t1"
        assert result.error == "interrupted"


# ── Metric events ───────────────────────────────────────────────


class TestMetricEvents:
    def test_metric_sampled(self):
        from skyward.actors.messages import Metric

        ni = _node_instance(4)
        ev = Metric(instance=ni, name="gpu_util", value=0.87, timestamp=0.0)
        result = translate(_spy(ev), "pool")

        assert isinstance(result, MetricEvent.Sampled)
        assert result.node_id == 4
        assert result.name == "gpu_util"
        assert result.value == 0.87

    def test_log_emitted(self):
        from skyward.actors.messages import Log

        ni = _node_instance(1)
        ev = Log(instance=ni, line="  epoch 1/10  ")
        result = translate(_spy(ev), "pool")

        assert isinstance(result, LogEvent.Emitted)
        assert result.node_id == 1
        assert result.message == "epoch 1/10"

    def test_log_empty_skipped(self):
        from skyward.actors.messages import Log

        ni = _node_instance(1)
        ev = Log(instance=ni, line="   ")
        result = translate(_spy(ev), "pool")

        assert result is None


# ── Pool lifecycle ──────────────────────────────────────────────


class TestPoolLifecycle:
    def test_provision_failed(self):
        from skyward.actors.pool.messages import ProvisionFailed

        ev = ProvisionFailed(reason="quota exceeded")
        result = translate(_spy(ev), "pool")

        assert isinstance(result, Pool.ProvisionFailed)
        assert result.reason == "quota exceeded"

    def test_shutdown_requested(self):
        from skyward.actors.messages import ShutdownRequested

        ev = ShutdownRequested(cluster_id="c1")
        result = translate(_spy(ev), "pool")

        assert isinstance(result, Pool.Stopped)
        assert result.pool_name == "pool"

    def test_stop_pool(self):
        from skyward.actors.pool.messages import StopPool

        ev = StopPool(reply_to=MagicMock())
        result = translate(_spy(ev), "pool")

        assert isinstance(result, Pool.Stopped)
        assert result.pool_name == "pool"

    def test_cluster_ready_is_noop(self):
        from skyward.actors.messages import ClusterReady

        mock_cluster = MagicMock()
        ev = ClusterReady(cluster=mock_cluster)
        result = translate(_spy(ev), "pool")

        assert result is None

    def test_instances_provisioned(self):
        from skyward.actors.pool.messages import InstancesProvisioned

        mock_cluster = MagicMock()
        mock_instances = (MagicMock(), MagicMock())
        ev = InstancesProvisioned(cluster=mock_cluster, instances=mock_instances)
        result = translate(_spy(ev), "pool")

        assert isinstance(result, Pool.Provisioned)
        assert result.cluster is mock_cluster
        assert result.instances is mock_instances


# ── Scaling events ──────────────────────────────────────────────


class TestScalingEvents:
    def test_desired_count_changed(self):
        from skyward.actors.messages import DesiredCountChanged

        ev = DesiredCountChanged(desired=6, reason="pressure")
        result = translate(_spy(ev), "pool")

        assert isinstance(result, Scaling.DesiredChanged)
        assert result.desired == 6
        assert result.reason == "pressure"

    def test_spawn_nodes(self):
        from skyward.actors.messages import SpawnNodes

        instances = ("i-1", "i-2", "i-3")
        ev = SpawnNodes(instances=instances, cluster=MagicMock(), start_node_id=0)
        result = translate(_spy(ev), "pool")

        assert isinstance(result, Scaling.Spawning)
        assert result.count == 3
        assert result.instances == instances

    def test_drain_node(self):
        from skyward.actors.messages import DrainNode

        ev = DrainNode(node_id=2, reply_to=MagicMock())
        result = translate(_spy(ev), "pool")

        assert isinstance(result, Scaling.Draining)
        assert result.node_id == 2

    def test_drain_complete(self):
        from skyward.actors.messages import DrainComplete

        ev = DrainComplete(node_id=2, instance_id="i-123")
        result = translate(_spy(ev), "pool")

        assert isinstance(result, Scaling.DrainCompleted)
        assert result.node_id == 2


# ── Error events ────────────────────────────────────────────────


class TestErrorEvents:
    def test_error_occurred(self):
        from skyward.actors.messages import Error

        ev = Error(request_id="r1", message="something broke", fatal=True)
        result = translate(_spy(ev), "pool")

        assert isinstance(result, ErrorEvent.Occurred)
        assert result.message == "something broke"
        assert result.fatal is True

    def test_error_non_fatal(self):
        from skyward.actors.messages import Error

        ev = Error(request_id="r1", message="minor issue")
        result = translate(_spy(ev), "pool")

        assert isinstance(result, ErrorEvent.Occurred)
        assert result.fatal is False


# ── No-ops ──────────────────────────────────────────────────────


class TestNoOps:
    def test_terminated_returns_none(self):
        from casty import Terminated

        result = translate(_spy(Terminated(ref=MagicMock())), "pool")
        assert result is None

    def test_pool_stopped_returns_none(self):
        from skyward.actors.pool.messages import PoolStopped

        result = translate(_spy(PoolStopped()), "pool")
        assert result is None

    def test_shutdown_done_returns_none(self):
        from skyward.actors.pool.messages import _ShutdownDone

        result = translate(_spy(_ShutdownDone()), "pool")
        assert result is None

    def test_provision_returns_none(self):
        from skyward.actors.messages import Provision

        ev = Provision(cluster=MagicMock(), provider=MagicMock(), instance=MagicMock())
        result = translate(_spy(ev), "pool")
        assert result is None

    def test_node_became_ready_returns_none(self):
        from skyward.actors.messages import NodeBecameReady

        ev = NodeBecameReady(node_id=0, instance=MagicMock())
        result = translate(_spy(ev), "pool")
        assert result is None

    def test_node_activated_returns_none(self):
        from skyward.actors.messages import NodeActivated

        ev = NodeActivated(node_id=0, node_ref=MagicMock(), slots=1)
        result = translate(_spy(ev), "pool")
        assert result is None

    def test_node_connected_returns_none(self):
        from skyward.actors.messages import NodeConnected

        ev = NodeConnected(node_id=0, instance=MagicMock())
        result = translate(_spy(ev), "pool")
        assert result is None

    def test_poll_result_returns_none(self):
        from skyward.actors.node.messages import _PollResult

        result = translate(_spy(_PollResult()), "pool")
        assert result is None

    def test_execute_on_node_returns_none(self):
        from skyward.actors.messages import ExecuteOnNode

        ev = ExecuteOnNode(fn=lambda: None, args=(), kwargs={}, reply_to=MagicMock())
        result = translate(_spy(ev), "pool")
        assert result is None

    def test_local_install_done_returns_none(self):
        from skyward.actors.node.messages import _LocalInstallDone

        ev = _LocalInstallDone(instance=MagicMock())
        result = translate(_spy(ev), "pool")
        assert result is None

    def test_user_code_sync_done_returns_none(self):
        from skyward.actors.node.messages import _UserCodeSyncDone

        ev = _UserCodeSyncDone(instance=MagicMock())
        result = translate(_spy(ev), "pool")
        assert result is None

    def test_node_joined_returns_none(self):
        from skyward.actors.messages import NodeJoined

        ev = NodeJoined(node_id=0)
        result = translate(_spy(ev), "pool")
        assert result is None

    def test_reconciler_node_lost_returns_none(self):
        from skyward.actors.messages import ReconcilerNodeLost

        ev = ReconcilerNodeLost(node_id=0, reason="x")
        result = translate(_spy(ev), "pool")
        assert result is None

    def test_unknown_event_returns_none(self):
        result = translate(_spy("some random string"), "pool")
        assert result is None


# ── Helpers ─────────────────────────────────────────────────────


class TestNodeIdFromPath:
    def test_extracts_node_id(self):
        assert _node_id_from_path("/system/pool/node-3") == 3

    def test_extracts_from_nested_path(self):
        assert _node_id_from_path("/system/pool/node-12/transport") == 12

    def test_returns_none_without_match(self):
        assert _node_id_from_path("/system/pool") is None


class TestFormatTask:
    def test_simple(self):
        def train(): ...

        result = _format_task(train, (1, 2), {"lr": 0.01})
        assert result == "train(1, 2, lr=0.01)"

    def test_no_args(self):
        def train(): ...

        result = _format_task(train, (), {})
        assert result == "train()"

    def test_truncation(self):
        def train(): ...

        long_arg = "x" * 100
        result = _format_task(train, (long_arg,), {})
        assert len(result) <= 81  # 80 + ellipsis char
        assert result.endswith("\u2026")

    def test_non_callable_uses_str(self):
        result = _format_task(42, (), {})
        assert result == "42()"
