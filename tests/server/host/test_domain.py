"""Tests for server domain ADTs."""

from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from skyward.api.spec import Nodes, Spec
from skyward.providers.container.config import Container
from skyward.server.host.domain import (
    Blob,
    Broadcast,
    CancelledExec,
    Compute,
    ComputeSpec,
    ComputeStatus,
    Dispatching,
    Error,
    Failed,
    FailedExec,
    FailedRes,
    GroupMember,
    InterruptedExec,
    InterruptedRes,
    Node,
    NodeBootstrapping,
    NodeConnecting,
    NodeLost,
    NodeReady,
    NodeStatus,
    NodeWaiting,
    PendingRes,
    Provider,
    Provisioning,
    Queued,
    Ready,
    ResultStatus,
    Run,
    RunningExec,
    RunningRes,
    Stopped,
    Stopping,
    SucceededExec,
    SucceededRes,
    Task,
    TaskExecution,
    TaskExecutionKind,
    TaskResult,
)


def _now() -> datetime:
    return datetime(2026, 4, 14, 12, 0, 0, tzinfo=timezone.utc)


def _spec() -> Spec:
    return Spec(provider=Container())


def _compute_spec() -> ComputeSpec:
    return ComputeSpec(
        specs=(_spec(),),
        selection="cheapest",
        nodes=Nodes(desired=2),
        allocation="spot-if-available",
        ttl=timedelta(hours=1),
    )


class TestComputeStatus:
    def test_each_variant_constructible_with_design_fields(self) -> None:
        t = _now()
        provisioning = Provisioning(started_at=t)
        ready = Ready(started_at=t, chosen=_spec(), nodes_ready=4, last_activity_at=t)
        stopping = Stopping(started_at=t, stopping_since=t)
        stopped = Stopped(started_at=t, stopped_at=t)
        failed = Failed(failed_at=t, reason="boom")

        assert provisioning.started_at == t
        assert ready.nodes_ready == 4
        assert stopping.stopping_since == t
        assert stopped.stopped_at == t
        assert failed.reason == "boom"

    def test_match_dispatch(self) -> None:
        t = _now()
        statuses: list[ComputeStatus] = [
            Provisioning(started_at=t),
            Ready(started_at=t, chosen=_spec(), nodes_ready=1, last_activity_at=t),
            Stopping(started_at=t, stopping_since=t),
            Stopped(started_at=t, stopped_at=t),
            Failed(failed_at=t, reason="x"),
        ]
        labels: list[str] = []
        for status in statuses:
            match status:
                case Provisioning():
                    labels.append("provisioning")
                case Ready(nodes_ready=n):
                    labels.append(f"ready:{n}")
                case Stopping():
                    labels.append("stopping")
                case Stopped():
                    labels.append("stopped")
                case Failed(reason=r):
                    labels.append(f"failed:{r}")

        assert labels == ["provisioning", "ready:1", "stopping", "stopped", "failed:x"]

    def test_replace_round_trip(self) -> None:
        t = _now()
        ready = Ready(started_at=t, chosen=_spec(), nodes_ready=1, last_activity_at=t)
        grown = replace(ready, nodes_ready=8)
        assert grown.nodes_ready == 8
        assert grown.started_at == ready.started_at

    def test_frozen(self) -> None:
        with pytest.raises(FrozenInstanceError):
            Provisioning(started_at=_now()).started_at = _now()  # type: ignore[misc]


class TestNodeStatus:
    def test_each_variant_constructible(self) -> None:
        t = _now()
        waiting = NodeWaiting()
        connecting = NodeConnecting(since=t)
        bootstrapping = NodeBootstrapping(since=t, phase="apt")
        ready = NodeReady(since=t)
        lost = NodeLost(at=t, reason="preempted")

        assert isinstance(waiting, NodeWaiting)
        assert connecting.since == t
        assert bootstrapping.phase == "apt"
        assert ready.since == t
        assert lost.reason == "preempted"

    def test_match_dispatch(self) -> None:
        t = _now()
        statuses: list[NodeStatus] = [
            NodeWaiting(),
            NodeConnecting(since=t),
            NodeBootstrapping(since=t, phase="pip"),
            NodeReady(since=t),
            NodeLost(at=t, reason="oom"),
        ]
        labels: list[str] = []
        for status in statuses:
            match status:
                case NodeWaiting():
                    labels.append("waiting")
                case NodeConnecting():
                    labels.append("connecting")
                case NodeBootstrapping(phase=p):
                    labels.append(f"boot:{p}")
                case NodeReady():
                    labels.append("ready")
                case NodeLost(reason=r):
                    labels.append(f"lost:{r}")

        assert labels == ["waiting", "connecting", "boot:pip", "ready", "lost:oom"]

    def test_frozen(self) -> None:
        with pytest.raises(FrozenInstanceError):
            NodeReady(since=_now()).since = _now()  # type: ignore[misc]


class TestTaskExecutionKind:
    def test_each_variant_constructible(self) -> None:
        run = Run()
        broadcast = Broadcast()
        member = GroupMember(group="g-1")

        assert isinstance(run, Run)
        assert isinstance(broadcast, Broadcast)
        assert member.group == "g-1"

    def test_match_dispatch(self) -> None:
        kinds: list[TaskExecutionKind] = [Run(), Broadcast(), GroupMember(group="g-2")]
        labels: list[str] = []
        for kind in kinds:
            match kind:
                case Run():
                    labels.append("run")
                case Broadcast():
                    labels.append("broadcast")
                case GroupMember(group=g):
                    labels.append(f"group:{g}")
        assert labels == ["run", "broadcast", "group:g-2"]

    def test_frozen(self) -> None:
        with pytest.raises(FrozenInstanceError):
            GroupMember(group="g").group = "h"  # type: ignore[misc]


class TestExecutionStatus:
    def test_each_variant_constructible(self) -> None:
        t = _now()
        queued = Queued()
        dispatching = Dispatching()
        running = RunningExec()
        succeeded = SucceededExec(finished_at=t)
        failed = FailedExec(finished_at=t)
        interrupted = InterruptedExec(interrupted_at=t, reason="server_restart")
        cancelled = CancelledExec(cancelled_at=t, reason="user")

        assert isinstance(queued, Queued)
        assert isinstance(dispatching, Dispatching)
        assert isinstance(running, RunningExec)
        assert succeeded.finished_at == t
        assert failed.finished_at == t
        assert interrupted.reason == "server_restart"
        assert cancelled.reason == "user"

    def test_match_dispatch(self) -> None:
        t = _now()
        labels: list[str] = []
        for status in (
            Queued(),
            Dispatching(),
            RunningExec(),
            SucceededExec(finished_at=t),
            FailedExec(finished_at=t),
            InterruptedExec(interrupted_at=t, reason="r"),
            CancelledExec(cancelled_at=t, reason="c"),
        ):
            match status:
                case Queued():
                    labels.append("queued")
                case Dispatching():
                    labels.append("dispatching")
                case RunningExec():
                    labels.append("running")
                case SucceededExec():
                    labels.append("succeeded")
                case FailedExec():
                    labels.append("failed")
                case InterruptedExec(reason=r):
                    labels.append(f"int:{r}")
                case CancelledExec(reason=r):
                    labels.append(f"cancel:{r}")
        assert labels == [
            "queued",
            "dispatching",
            "running",
            "succeeded",
            "failed",
            "int:r",
            "cancel:c",
        ]

    def test_frozen(self) -> None:
        with pytest.raises(FrozenInstanceError):
            SucceededExec(finished_at=_now()).finished_at = _now()  # type: ignore[misc]


class TestResultStatus:
    def test_each_variant_constructible(self) -> None:
        t = _now()
        pending = PendingRes()
        running = RunningRes(dispatched_at=t, started_at=t, node="n-1")
        succeeded = SucceededRes(
            dispatched_at=t, started_at=t, finished_at=t, node="n-1", blob=42
        )
        failed = FailedRes(
            dispatched_at=t, started_at=None, finished_at=t, node="n-1", error=7
        )
        interrupted = InterruptedRes(
            dispatched_at=t,
            started_at=None,
            interrupted_at=t,
            node="n-1",
            reason="lost",
        )

        assert isinstance(pending, PendingRes)
        assert running.node == "n-1"
        assert succeeded.blob == 42
        assert failed.error == 7
        assert interrupted.reason == "lost"

    def test_match_dispatch(self) -> None:
        t = _now()
        statuses: list[ResultStatus] = [
            PendingRes(),
            RunningRes(dispatched_at=t, started_at=t, node="n"),
            SucceededRes(
                dispatched_at=t, started_at=t, finished_at=t, node="n", blob=1
            ),
            FailedRes(
                dispatched_at=t, started_at=t, finished_at=t, node="n", error=2
            ),
            InterruptedRes(
                dispatched_at=t,
                started_at=t,
                interrupted_at=t,
                node="n",
                reason="r",
            ),
        ]
        labels: list[str] = []
        for status in statuses:
            match status:
                case PendingRes():
                    labels.append("pending")
                case RunningRes(node=n):
                    labels.append(f"running:{n}")
                case SucceededRes(blob=b):
                    labels.append(f"ok:{b}")
                case FailedRes(error=e):
                    labels.append(f"err:{e}")
                case InterruptedRes(reason=r):
                    labels.append(f"int:{r}")
        assert labels == ["pending", "running:n", "ok:1", "err:2", "int:r"]

    def test_frozen(self) -> None:
        with pytest.raises(FrozenInstanceError):
            RunningRes(dispatched_at=_now(), started_at=_now(), node="n").node = "x"  # type: ignore[misc]


class TestProductTypes:
    def test_provider_round_trip(self) -> None:
        t = _now()
        p = Provider(
            name="aws-default",
            type="aws",
            config={"region": "us-east-1"},
            created_at=t,
            updated_at=t,
            last_used_at=None,
        )
        bumped = replace(p, last_used_at=t)
        assert bumped.last_used_at == t
        assert bumped.config == {"region": "us-east-1"}

    def test_compute_round_trip(self) -> None:
        t = _now()
        c = Compute(
            name="my-pool",
            spec=_compute_spec(),
            created_at=t,
            status=Provisioning(started_at=t),
        )
        ready = replace(
            c,
            status=Ready(
                started_at=t, chosen=_spec(), nodes_ready=2, last_activity_at=t
            ),
        )
        match ready.status:
            case Ready(nodes_ready=n):
                assert n == 2
            case _:
                pytest.fail("expected Ready")

    def test_node_round_trip(self) -> None:
        t = _now()
        n = Node(
            id="n-1",
            compute="my-pool",
            instance_id="i-abc",
            provider_name="aws-default",
            head_addr=None,
            status=NodeWaiting(),
            created_at=t,
        )
        joined = replace(n, head_addr="10.0.0.1", status=NodeReady(since=t))
        assert joined.head_addr == "10.0.0.1"
        match joined.status:
            case NodeReady():
                pass
            case _:
                pytest.fail("expected NodeReady")

    def test_task_and_execution(self) -> None:
        t = _now()
        task = Task(module="my.mod", qualname="train")
        execution = TaskExecution(
            id="exec-1",
            task=(task.module, task.qualname),
            compute="my-pool",
            kind=Run(),
            payload=10,
            timeout=timedelta(minutes=5),
            client="client-a",
            submitted_at=t,
            status=Queued(),
        )
        finished = replace(execution, status=SucceededExec(finished_at=t))
        assert finished.task == ("my.mod", "train")
        assert finished.payload == 10
        match finished.status:
            case SucceededExec():
                pass
            case _:
                pytest.fail("expected SucceededExec")

    def test_task_result(self) -> None:
        result = TaskResult(id=1, execution="exec-1", shard=0, status=PendingRes())
        evolved = replace(
            result,
            status=SucceededRes(
                dispatched_at=_now(),
                started_at=_now(),
                finished_at=_now(),
                node="n-1",
                blob=99,
            ),
        )
        assert evolved.shard == 0
        match evolved.status:
            case SucceededRes(blob=b):
                assert b == 99
            case _:
                pytest.fail("expected SucceededRes")

    def test_blob(self) -> None:
        t = _now()
        blob = Blob(
            id=42,
            path=Path("/tmp/blobs/00000042.bin"),
            size=1024,
            sha256=None,
            kind="payload",
            created_at=t,
            evicted_at=None,
        )
        evicted = replace(blob, evicted_at=t)
        assert evicted.evicted_at == t
        assert evicted.kind == "payload"

    def test_error(self) -> None:
        t = _now()
        err = Error(
            id=7,
            type="ValueError",
            message="bad input",
            traceback=None,
            created_at=t,
        )
        with_tb = replace(err, traceback="line 1\nline 2")
        assert with_tb.traceback == "line 1\nline 2"

    def test_products_are_frozen(self) -> None:
        t = _now()
        task = Task(module="m", qualname="q")
        with pytest.raises(FrozenInstanceError):
            task.module = "x"  # type: ignore[misc]
        blob = Blob(
            id=1,
            path=Path("/x"),
            size=0,
            sha256=None,
            kind="result",
            created_at=t,
            evicted_at=None,
        )
        with pytest.raises(FrozenInstanceError):
            blob.size = 5  # type: ignore[misc]
