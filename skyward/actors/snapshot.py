from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from skyward.core.model import Cluster, Instance

__all__ = [
    "BootstrapTimeline",
    "NodeSnapshot",
    "NodeStatus",
    "PoolPhase",
    "PoolSnapshot",
    "ScalingSnapshot",
    "TaskCounters",
]


class PoolPhase(Enum):
    PROVISIONING = auto()
    SSH = auto()
    BOOTSTRAP = auto()
    WORKERS = auto()
    READY = auto()
    STOPPING = auto()


class NodeStatus(Enum):
    WAITING = auto()
    SSH = auto()
    BOOTSTRAPPING = auto()
    READY = auto()


@dataclass(frozen=True, slots=True)
class BootstrapTimeline:
    phases: tuple[str, ...]
    completed: frozenset[str]
    active: str
    output: str


@dataclass(frozen=True, slots=True)
class NodeSnapshot:
    node_id: int
    instance_id: str
    status: NodeStatus
    bootstrap: BootstrapTimeline | None = None


@dataclass(frozen=True, slots=True)
class TaskCounters:
    queued: int = 0
    running: int = 0
    done: int = 0
    failed: int = 0


@dataclass(frozen=True, slots=True)
class ScalingSnapshot:
    desired_nodes: int = 0
    pending_nodes: int = 0
    draining_nodes: int = 0
    reconciler_state: str = "watching"
    is_elastic: bool = False
    min_nodes: int | None = None
    max_nodes: int | None = None


@dataclass(frozen=True, slots=True)
class PoolSnapshot:
    name: str
    phase: PoolPhase
    nodes: tuple[NodeSnapshot, ...]
    tasks: TaskCounters
    scaling: ScalingSnapshot
    cluster: Cluster | None = None
    instances: tuple[Instance, ...] = ()
    started_at: float = 0.0
