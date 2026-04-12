"""Domain events ADT — the vocabulary that feeds SessionProjection.

Scala-style ADT: namespace classes group frozen dataclass events.
Every event carries ``pool_name`` so projections can dispatch by pool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from skyward.actors.snapshot import PoolSnapshot
    from skyward.api.model import Instance

__all__ = [
    "Error",
    "ErrorEvent",
    "Log",
    "LogEvent",
    "Metric",
    "MetricEvent",
    "Node",
    "NodeEvent",
    "Pool",
    "PoolEvent",
    "Scaling",
    "ScalingEvent",
    "SessionEvent",
    "Task",
    "TaskEvent",
]


class Pool:
    @dataclass(frozen=True, slots=True)
    class Provisioning:
        pool_name: str
        total_nodes: int
        started_at: float

    @dataclass(frozen=True, slots=True)
    class PhaseChanged:
        pool_name: str
        phase: str

    @dataclass(frozen=True, slots=True)
    class Stopped:
        pool_name: str

    @dataclass(frozen=True, slots=True)
    class Reconciled:
        pool_name: str
        snapshot: PoolSnapshot | None

    @dataclass(frozen=True, slots=True)
    class Provisioned:
        pool_name: str
        cluster: Any
        instances: tuple[Instance, ...]

    @dataclass(frozen=True, slots=True)
    class ProvisionFailed:
        pool_name: str
        reason: str


class Node:
    @dataclass(frozen=True, slots=True)
    class Connected:
        pool_name: str
        node_id: int
        instance: Instance | None

    @dataclass(frozen=True, slots=True)
    class Ready:
        pool_name: str
        node_id: int

    @dataclass(frozen=True, slots=True)
    class Lost:
        pool_name: str
        node_id: int
        reason: str
        instance_id: str | None = None

    @dataclass(frozen=True, slots=True)
    class ConnectionFailed:
        pool_name: str
        error: str

    @dataclass(frozen=True, slots=True)
    class Preempted:
        pool_name: str
        reason: str

    @dataclass(frozen=True, slots=True)
    class WorkerFailed:
        pool_name: str
        error: str

    class Bootstrap:
        @dataclass(frozen=True, slots=True)
        class Started:
            pool_name: str
            node_id: int
            phase: str

        @dataclass(frozen=True, slots=True)
        class Completed:
            pool_name: str
            node_id: int
            phase: str

        @dataclass(frozen=True, slots=True)
        class Output:
            pool_name: str
            node_id: int
            output: str
            overwrite: bool = False

        @dataclass(frozen=True, slots=True)
        class Done:
            pool_name: str
            node_id: int
            success: bool
            error: str | None = None

        @dataclass(frozen=True, slots=True)
        class Failed:
            pool_name: str
            node_id: int
            phase: str
            error: str

        @dataclass(frozen=True, slots=True)
        class Command:
            pool_name: str
            node_id: int
            command: str


class Task:
    @dataclass(frozen=True, slots=True)
    class Queued:
        pool_name: str
        task_id: str
        name: str
        kind: str
        broadcast_total: int = 0

    @dataclass(frozen=True, slots=True)
    class Assigned:
        pool_name: str
        task_id: str
        node_id: int

    @dataclass(frozen=True, slots=True)
    class Completed:
        pool_name: str
        task_id: str
        node_id: int
        elapsed: float

    @dataclass(frozen=True, slots=True)
    class Failed:
        pool_name: str
        task_id: str
        node_id: int
        error: str

    @dataclass(frozen=True, slots=True)
    class BroadcastPartial:
        pool_name: str
        task_id: str


class Metric:
    @dataclass(frozen=True, slots=True)
    class Sampled:
        pool_name: str
        node_id: int
        name: str
        value: float


class Log:
    @dataclass(frozen=True, slots=True)
    class Emitted:
        pool_name: str
        node_id: int
        message: str
        level: str = "info"
        overwrite: bool = False


class Scaling:
    @dataclass(frozen=True, slots=True)
    class DesiredChanged:
        pool_name: str
        desired: int
        reason: str

    @dataclass(frozen=True, slots=True)
    class Spawning:
        pool_name: str
        count: int
        instances: tuple[Any, ...] = ()

    @dataclass(frozen=True, slots=True)
    class Draining:
        pool_name: str
        count: int

    @dataclass(frozen=True, slots=True)
    class DrainCompleted:
        pool_name: str
        node_id: int


class Error:
    @dataclass(frozen=True, slots=True)
    class Occurred:
        pool_name: str
        message: str
        fatal: bool = False


# ── Type unions ──────────────────────────────────────────────────

type PoolEvent = (
    Pool.Provisioning
    | Pool.Provisioned
    | Pool.PhaseChanged
    | Pool.Stopped
    | Pool.Reconciled
    | Pool.ProvisionFailed
)

type NodeEvent = (
    Node.Connected
    | Node.Ready
    | Node.Lost
    | Node.ConnectionFailed
    | Node.Preempted
    | Node.WorkerFailed
    | Node.Bootstrap.Started
    | Node.Bootstrap.Completed
    | Node.Bootstrap.Output
    | Node.Bootstrap.Done
    | Node.Bootstrap.Failed
    | Node.Bootstrap.Command
)

type TaskEvent = (
    Task.Queued
    | Task.Assigned
    | Task.Completed
    | Task.Failed
    | Task.BroadcastPartial
)

type MetricEvent = Metric.Sampled

type LogEvent = Log.Emitted

type ScalingEvent = (
    Scaling.DesiredChanged
    | Scaling.Spawning
    | Scaling.Draining
    | Scaling.DrainCompleted
)

type ErrorEvent = Error.Occurred

type SessionEvent = (
    PoolEvent
    | NodeEvent
    | TaskEvent
    | MetricEvent
    | LogEvent
    | ScalingEvent
    | ErrorEvent
)
