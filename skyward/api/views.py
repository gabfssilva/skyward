"""Read-only view hierarchy for running session state.

Pure data — no Rich, no Casty, no spy.  These frozen dataclasses are
the agnostic representation that both the console actor and the CLI
read from a ``SessionProjection``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skyward.api.model import Cluster, Instance
    from skyward.api.spec import PoolSpec

__all__ = [
    "BootstrapView",
    "NodeStatus",
    "NodeView",
    "PoolPhase",
    "PoolView",
    "ScalingView",
    "SessionView",
    "TaskEntry",
    "TasksView",
]

# ── Sentinel empty maps (shared across all default instances) ────

_EMPTY_MAP: MappingProxyType = MappingProxyType({})
_EMPTY_FLOAT_MAP: MappingProxyType[str, float] = MappingProxyType({})
_EMPTY_LATENCIES: MappingProxyType[str, tuple[float, ...]] = MappingProxyType({})
_EMPTY_FAILED: MappingProxyType[str, int] = MappingProxyType({})
_EMPTY_NODE_TASKS: MappingProxyType[int, int] = MappingProxyType({})

# ── Enums ────────────────────────────────────────────────────────


class PoolPhase(Enum):
    PROVISIONING = auto()
    SSH = auto()
    BOOTSTRAP = auto()
    WORKERS = auto()
    READY = auto()
    STOPPED = auto()


class NodeStatus(Enum):
    WAITING = auto()
    SSH = auto()
    BOOTSTRAPPING = auto()
    READY = auto()


# ── Node-level views ────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class BootstrapView:
    phases: tuple[str, ...]
    completed: frozenset[str]
    active: str
    output: str


@dataclass(frozen=True, slots=True)
class NodeView:
    node_id: int
    status: NodeStatus
    instance: Instance | None = None
    bootstrap: BootstrapView | None = None
    metrics: MappingProxyType[str, float] = field(
        default_factory=lambda: _EMPTY_FLOAT_MAP
    )


# ── Task-level views ────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TaskEntry:
    task_id: str
    name: str
    kind: str
    started_at: float
    node_id: int = -1
    broadcast_total: int = 0
    broadcast_done: int = 0


@dataclass(frozen=True, slots=True)
class TasksView:
    queued: int = 0
    running: int = 0
    done: int = 0
    failed: int = 0
    inflight: MappingProxyType[str, TaskEntry] = field(
        default_factory=lambda: _EMPTY_MAP
    )
    latencies: tuple[float, ...] = ()
    fn_stats: MappingProxyType[str, tuple[float, ...]] = field(
        default_factory=lambda: _EMPTY_LATENCIES
    )
    fn_failed: MappingProxyType[str, int] = field(
        default_factory=lambda: _EMPTY_FAILED
    )
    first_task_at: float = 0.0
    throughput: float = 0.0
    tasks_per_node: MappingProxyType[int, int] = field(
        default_factory=lambda: _EMPTY_NODE_TASKS
    )


# ── Scaling view ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ScalingView:
    desired: int = 0
    pending: int = 0
    draining: int = 0
    reconciler_state: str = "watching"
    is_elastic: bool = False
    min_nodes: int | None = None
    max_nodes: int | None = None


# ── Pool-level view ─────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PoolView:
    name: str
    phase: PoolPhase
    tasks: TasksView
    scaling: ScalingView
    total_nodes: int = 0
    nodes: MappingProxyType[int, NodeView] = field(
        default_factory=lambda: _EMPTY_MAP
    )
    cluster: Cluster | None = None
    instances: tuple[Instance, ...] = ()
    started_at: float = 0.0
    ready_at: float = 0.0
    spec: PoolSpec | None = None


# ── Session-level view ──────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SessionView:
    pools: MappingProxyType[str, PoolView] = field(
        default_factory=lambda: _EMPTY_MAP
    )
