"""Server domain ADTs.

Canonical state for the HTTP server. Sum types model lifecycle status
(``ComputeStatus``, ``NodeStatus``, ``ExecutionStatus``, ``ResultStatus``,
``TaskExecutionKind``); product types model entities (``Provider``,
``Compute``, ``Node``, ``Task``, ``TaskExecution``, ``TaskResult``,
``Blob``, ``Error``).
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from skyward.api.spec import Nodes, Spec

type ComputeName = str
type TaskKey = tuple[str, str]
type ExecutionId = str
type ResultId = int
type GroupId = str
type NodeId = str
type BlobId = int
type ErrorId = int
type ClientId = str

type ProviderType = Literal[
    "aws",
    "gcp",
    "vastai",
    "runpod",
    "hyperstack",
    "lambda",
    "tensordock",
    "thunder",
    "verda",
    "vultr",
    "novita",
    "scaleway",
    "massedcompute",
    "jarvislabs",
    "container",
]
type SelectionStrategy = Literal["cheapest", "first"]
type AllocationStrategy = Literal[
    "spot", "on-demand", "spot-if-available", "cheapest"
]
type BlobKind = Literal["payload", "result"]


@dataclass(frozen=True, slots=True)
class Provider:
    name: str
    type: ProviderType
    config: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    last_used_at: datetime | None


@dataclass(frozen=True, slots=True)
class ComputeSpec:
    specs: tuple[Spec, ...]
    selection: SelectionStrategy
    nodes: Nodes
    allocation: AllocationStrategy
    ttl: timedelta


@dataclass(frozen=True, slots=True)
class Provisioning:
    started_at: datetime


@dataclass(frozen=True, slots=True)
class Ready:
    started_at: datetime
    chosen: Spec
    nodes_ready: int
    last_activity_at: datetime


@dataclass(frozen=True, slots=True)
class Stopping:
    started_at: datetime
    stopping_since: datetime


@dataclass(frozen=True, slots=True)
class Stopped:
    started_at: datetime
    stopped_at: datetime


@dataclass(frozen=True, slots=True)
class Failed:
    failed_at: datetime
    reason: str


type ComputeStatus = Provisioning | Ready | Stopping | Stopped | Failed


@dataclass(frozen=True, slots=True)
class Compute:
    name: ComputeName
    spec: ComputeSpec
    created_at: datetime
    status: ComputeStatus


@dataclass(frozen=True, slots=True)
class NodeWaiting:
    pass


@dataclass(frozen=True, slots=True)
class NodeConnecting:
    since: datetime


@dataclass(frozen=True, slots=True)
class NodeBootstrapping:
    since: datetime
    phase: str


@dataclass(frozen=True, slots=True)
class NodeReady:
    since: datetime


@dataclass(frozen=True, slots=True)
class NodeLost:
    at: datetime
    reason: str


type NodeStatus = (
    NodeWaiting | NodeConnecting | NodeBootstrapping | NodeReady | NodeLost
)


@dataclass(frozen=True, slots=True)
class Node:
    id: NodeId
    compute: ComputeName
    instance_id: str
    provider_name: str
    head_addr: str | None
    status: NodeStatus
    created_at: datetime


@dataclass(frozen=True, slots=True)
class Task:
    module: str
    qualname: str


@dataclass(frozen=True, slots=True)
class Run:
    pass


@dataclass(frozen=True, slots=True)
class Broadcast:
    pass


@dataclass(frozen=True, slots=True)
class GroupMember:
    group: GroupId


type TaskExecutionKind = Run | Broadcast | GroupMember


@dataclass(frozen=True, slots=True)
class Queued:
    pass


@dataclass(frozen=True, slots=True)
class Dispatching:
    pass


@dataclass(frozen=True, slots=True)
class RunningExec:
    pass


@dataclass(frozen=True, slots=True)
class SucceededExec:
    finished_at: datetime


@dataclass(frozen=True, slots=True)
class FailedExec:
    finished_at: datetime


@dataclass(frozen=True, slots=True)
class InterruptedExec:
    interrupted_at: datetime
    reason: str


@dataclass(frozen=True, slots=True)
class CancelledExec:
    cancelled_at: datetime
    reason: str


type ExecutionStatus = (
    Queued
    | Dispatching
    | RunningExec
    | SucceededExec
    | FailedExec
    | InterruptedExec
    | CancelledExec
)


@dataclass(frozen=True, slots=True)
class TaskExecution:
    id: ExecutionId
    task: TaskKey
    compute: ComputeName
    kind: TaskExecutionKind
    payload: BlobId
    timeout: timedelta | None
    client: ClientId | None
    submitted_at: datetime
    status: ExecutionStatus


@dataclass(frozen=True, slots=True)
class PendingRes:
    pass


@dataclass(frozen=True, slots=True)
class RunningRes:
    dispatched_at: datetime
    started_at: datetime
    node: NodeId


@dataclass(frozen=True, slots=True)
class SucceededRes:
    dispatched_at: datetime
    started_at: datetime
    finished_at: datetime
    node: NodeId
    blob: BlobId


@dataclass(frozen=True, slots=True)
class FailedRes:
    dispatched_at: datetime
    started_at: datetime | None
    finished_at: datetime
    node: NodeId
    error: ErrorId


@dataclass(frozen=True, slots=True)
class InterruptedRes:
    dispatched_at: datetime
    started_at: datetime | None
    interrupted_at: datetime
    node: NodeId
    reason: str


type ResultStatus = (
    PendingRes | RunningRes | SucceededRes | FailedRes | InterruptedRes
)


@dataclass(frozen=True, slots=True)
class TaskResult:
    id: ResultId
    execution: ExecutionId
    shard: int
    status: ResultStatus


@dataclass(frozen=True, slots=True)
class Blob:
    id: BlobId
    path: Path
    size: int
    sha256: str | None
    kind: BlobKind
    created_at: datetime
    evicted_at: datetime | None


@dataclass(frozen=True, slots=True)
class Error:
    id: ErrorId
    type: str
    message: str
    traceback: str | None
    created_at: datetime


__all__ = [
    "AllocationStrategy",
    "Blob",
    "BlobId",
    "BlobKind",
    "Broadcast",
    "CancelledExec",
    "ClientId",
    "Compute",
    "ComputeName",
    "ComputeSpec",
    "ComputeStatus",
    "Dispatching",
    "Error",
    "ErrorId",
    "ExecutionId",
    "ExecutionStatus",
    "Failed",
    "FailedExec",
    "FailedRes",
    "GroupId",
    "GroupMember",
    "InterruptedExec",
    "InterruptedRes",
    "Node",
    "NodeBootstrapping",
    "NodeConnecting",
    "NodeId",
    "NodeLost",
    "NodeReady",
    "NodeStatus",
    "NodeWaiting",
    "PendingRes",
    "Provider",
    "ProviderType",
    "Provisioning",
    "Queued",
    "Ready",
    "ResultId",
    "ResultStatus",
    "Run",
    "RunningExec",
    "RunningRes",
    "SelectionStrategy",
    "Stopped",
    "Stopping",
    "SucceededExec",
    "SucceededRes",
    "Task",
    "TaskExecution",
    "TaskExecutionKind",
    "TaskKey",
    "TaskResult",
]
