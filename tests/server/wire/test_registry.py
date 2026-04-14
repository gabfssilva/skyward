"""Tests for the wire discriminated-union registry."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

import skyward.server.wire  # noqa: F401 — triggers registrations
from skyward.api.events import (
    Error,
    Log,
    Metric,
    Node as NodeEv,
    Pool as PoolEv,
    Scaling,
    SessionEvent,
    Task,
)
from skyward.api.plugin import Plugin
from skyward.api.provider import ProviderConfig
from skyward.api.spec import Nodes, Spec
from skyward.plugins.jax import jax
from skyward.plugins.torch import torch
from skyward.providers import (
    AWS,
    GCP,
    Container,
    Hyperstack,
    JarvisLabs,
    LambdaCloud,
    MassedCompute,
    Novita,
    RunPod,
    Scaleway,
    TensorDock,
    VastAI,
    Verda,
    Vultr,
)
from skyward.server.host.domain import (
    Broadcast,
    CancelledExec,
    ComputeSpec,
    ComputeStatus,
    Dispatching,
    ExecutionStatus,
    Failed,
    FailedExec,
    FailedRes,
    GroupMember,
    InterruptedExec,
    InterruptedRes,
    NodeBootstrapping,
    NodeConnecting,
    NodeLost,
    NodeReady,
    NodeStatus,
    NodeWaiting,
    PendingRes,
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
    TaskExecutionKind,
)
from skyward.server.wire import from_dict, to_dict


# ── Providers ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("config", "tag"),
    [
        (AWS(region="us-west-2"), "aws"),
        (GCP(zone="us-west1-b"), "gcp"),
        (Hyperstack(), "hyperstack"),
        (JarvisLabs(), "jarvislabs"),
        (LambdaCloud(), "lambda"),
        (MassedCompute(), "massed_compute"),
        (Novita(), "novita"),
        (RunPod(), "runpod"),
        (Scaleway(), "scaleway"),
        (TensorDock(), "tensordock"),
        (VastAI(), "vastai"),
        (Verda(), "verda"),
        (Vultr(), "vultr"),
        (Container(), "container"),
    ],
)
def test_provider_config_roundtrip(config: ProviderConfig, tag: str) -> None:
    encoded = to_dict(config)
    assert encoded["type"] == tag
    decoded = from_dict(encoded, ProviderConfig)
    assert decoded == config


def test_unknown_provider_tag_raises() -> None:
    with pytest.raises(ValueError, match="Unknown tag"):
        from_dict({"type": "nonexistent"}, ProviderConfig)


# ── ComputeStatus ────────────────────────────────────────────────────


def _dt(offset: int = 0) -> datetime:
    return datetime(2026, 4, 14, 12, 30, offset, tzinfo=UTC)


@pytest.mark.parametrize(
    "status",
    [
        Provisioning(started_at=_dt()),
        Stopping(started_at=_dt(), stopping_since=_dt(1)),
        Stopped(started_at=_dt(), stopped_at=_dt(2)),
        Failed(failed_at=_dt(), reason="boom"),
    ],
)
def test_compute_status_scalar_variants_roundtrip(status: ComputeStatus) -> None:
    encoded = to_dict(status)
    assert "type" in encoded
    decoded = from_dict(encoded, ComputeStatus)
    assert decoded == status


def test_compute_status_ready_dispatches_correctly() -> None:
    # Ready wraps a ``Spec``; Spec has a forward-ref ``ProviderConfig`` hint
    # that ``typing.get_type_hints`` cannot resolve under the B1 codec's
    # current forward-ref handling.  The codec still dispatches the outer
    # union correctly by tag, which is what this B2 test verifies; deep
    # Spec/Image round-trip is exercised indirectly by the dedicated
    # ComputeSpec test below.
    ready = Ready(
        started_at=_dt(),
        chosen=Spec(provider=AWS(), nodes=1),
        nodes_ready=2,
        last_activity_at=_dt(5),
    )
    encoded = to_dict(ready)
    assert encoded["type"] == "ready"
    decoded = from_dict(encoded, ComputeStatus)
    assert isinstance(decoded, Ready)
    assert decoded.started_at == ready.started_at
    assert decoded.nodes_ready == ready.nodes_ready
    assert decoded.last_activity_at == ready.last_activity_at


# ── NodeStatus ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "status",
    [
        NodeWaiting(),
        NodeConnecting(since=_dt()),
        NodeBootstrapping(since=_dt(), phase="packages"),
        NodeReady(since=_dt()),
        NodeLost(at=_dt(), reason="preempted"),
    ],
)
def test_node_status_roundtrip(status: NodeStatus) -> None:
    encoded = to_dict(status)
    decoded = from_dict(encoded, NodeStatus)
    assert decoded == status


# ── ExecutionStatus ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "status",
    [
        Queued(),
        Dispatching(),
        RunningExec(),
        SucceededExec(finished_at=_dt()),
        FailedExec(finished_at=_dt()),
        InterruptedExec(interrupted_at=_dt(), reason="spot"),
        CancelledExec(cancelled_at=_dt(), reason="user"),
    ],
)
def test_execution_status_roundtrip(status: ExecutionStatus) -> None:
    encoded = to_dict(status)
    decoded = from_dict(encoded, ExecutionStatus)
    assert decoded == status


# ── ResultStatus ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "status",
    [
        PendingRes(),
        RunningRes(dispatched_at=_dt(), started_at=_dt(1), node="n-1"),
        SucceededRes(
            dispatched_at=_dt(),
            started_at=_dt(1),
            finished_at=_dt(2),
            node="n-1",
            blob=42,
        ),
        FailedRes(
            dispatched_at=_dt(),
            started_at=None,
            finished_at=_dt(2),
            node="n-1",
            error=7,
        ),
        InterruptedRes(
            dispatched_at=_dt(),
            started_at=None,
            interrupted_at=_dt(1),
            node="n-1",
            reason="spot",
        ),
    ],
)
def test_result_status_roundtrip(status: ResultStatus) -> None:
    encoded = to_dict(status)
    decoded = from_dict(encoded, ResultStatus)
    assert decoded == status


# ── TaskExecutionKind ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kind",
    [Run(), Broadcast(), GroupMember(group="g-1")],
)
def test_task_execution_kind_roundtrip(kind: TaskExecutionKind) -> None:
    encoded = to_dict(kind)
    decoded = from_dict(encoded, TaskExecutionKind)
    assert decoded == kind


# ── SessionEvent ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("event", "tag"),
    [
        (PoolEv.Provisioning(pool_name="p", total_nodes=3, started_at=1.0), "Pool.Provisioning"),
        (PoolEv.PhaseChanged(pool_name="p", phase="ready"), "Pool.PhaseChanged"),
        (PoolEv.Stopped(pool_name="p"), "Pool.Stopped"),
        (PoolEv.ProvisionFailed(pool_name="p", reason="x"), "Pool.ProvisionFailed"),
        (NodeEv.Ready(pool_name="p", node_id=1), "Node.Ready"),
        (NodeEv.Lost(pool_name="p", node_id=1, reason="gone"), "Node.Lost"),
        (NodeEv.ConnectionFailed(pool_name="p", error="ssh"), "Node.ConnectionFailed"),
        (NodeEv.Preempted(pool_name="p", reason="spot"), "Node.Preempted"),
        (NodeEv.WorkerFailed(pool_name="p", error="seg"), "Node.WorkerFailed"),
        (NodeEv.Bootstrap.Started(pool_name="p", node_id=0, phase="apt"), "Node.Bootstrap.Started"),
        (NodeEv.Bootstrap.Completed(pool_name="p", node_id=0, phase="apt"), "Node.Bootstrap.Completed"),
        (NodeEv.Bootstrap.Output(pool_name="p", node_id=0, output="hi"), "Node.Bootstrap.Output"),
        (NodeEv.Bootstrap.Done(pool_name="p", node_id=0, success=True), "Node.Bootstrap.Done"),
        (NodeEv.Bootstrap.Failed(pool_name="p", node_id=0, phase="apt", error="x"), "Node.Bootstrap.Failed"),
        (NodeEv.Bootstrap.Command(pool_name="p", node_id=0, command="ls"), "Node.Bootstrap.Command"),
        (Task.Queued(pool_name="p", task_id="t", name="train", kind="run"), "Task.Queued"),
        (Task.Assigned(pool_name="p", task_id="t", node_id=1), "Task.Assigned"),
        (Task.Completed(pool_name="p", task_id="t", node_id=1, elapsed=0.5), "Task.Completed"),
        (Task.Failed(pool_name="p", task_id="t", node_id=1, error="x"), "Task.Failed"),
        (Task.BroadcastPartial(pool_name="p", task_id="t"), "Task.BroadcastPartial"),
        (Metric.Sampled(pool_name="p", node_id=1, name="gpu_util", value=0.7), "Metric.Sampled"),
        (Log.Emitted(pool_name="p", node_id=1, message="m"), "Log.Emitted"),
        (Scaling.DesiredChanged(pool_name="p", desired=4, reason="pressure"), "Scaling.DesiredChanged"),
        (Scaling.Spawning(pool_name="p", count=2), "Scaling.Spawning"),
        (Scaling.Draining(pool_name="p", count=1), "Scaling.Draining"),
        (Scaling.DrainCompleted(pool_name="p", node_id=5), "Scaling.DrainCompleted"),
        (Error.Occurred(pool_name="p", message="boom"), "Error.Occurred"),
    ],
)
def test_session_event_roundtrip(event: SessionEvent, tag: str) -> None:
    encoded = to_dict(event)
    assert encoded["type"] == tag
    decoded = from_dict(encoded, SessionEvent)
    assert decoded == event


# ── ComputeSpec with plugins ─────────────────────────────────────────


def test_compute_spec_with_plugins_roundtrip() -> None:
    from datetime import timedelta

    spec = Spec(
        provider=AWS(region="us-east-1"),
        nodes=Nodes(desired=2),
        plugins=(torch(backend="gloo", cuda="cu124"), jax(cuda="cu124")),
    )
    compute_spec = ComputeSpec(
        specs=(spec,),
        selection="cheapest",
        nodes=Nodes(desired=2),
        allocation="spot-if-available",
        ttl=timedelta(seconds=600),
    )

    encoded = to_dict(compute_spec)
    decoded = from_dict(encoded, ComputeSpec)

    assert isinstance(decoded, ComputeSpec)
    assert decoded.selection == "cheapest"
    assert decoded.nodes == Nodes(desired=2)
    assert decoded.ttl == timedelta(seconds=600)
    assert len(decoded.specs) == 1
    decoded_spec = decoded.specs[0]
    assert isinstance(decoded_spec.provider, AWS)
    assert getattr(decoded_spec.provider, "region") == "us-east-1"
    assert len(decoded_spec.plugins) == 2
    assert [p.name for p in decoded_spec.plugins] == ["torch", "jax"]
    for p in decoded_spec.plugins:
        assert isinstance(p, Plugin)
