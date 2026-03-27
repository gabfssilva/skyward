from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from types import MappingProxyType

from skyward.core.model import Cluster, Instance


class _Phase(Enum):
    PROVISIONING = auto()
    SSH = auto()
    BOOTSTRAP = auto()
    WORKERS = auto()
    READY = auto()
    STOPPING = auto()


class _NodeStatus(Enum):
    WAITING = auto()
    SSH = auto()
    BOOTSTRAPPING = auto()
    READY = auto()


@dataclass(frozen=True, slots=True)
class _BootstrapTimeline:
    phases: tuple[str, ...]
    completed: frozenset[str]
    active: str
    output: str


@dataclass(frozen=True, slots=True)
class _TaskEntry:
    task_id: str
    name: str
    kind: str
    started_at: float
    node_id: int = -1
    broadcast_total: int = 0
    broadcast_done: int = 0


@dataclass(frozen=True, slots=True)
class _State:
    total_nodes: int
    phase: _Phase = _Phase.PROVISIONING
    nodes: MappingProxyType[int, _NodeStatus] = MappingProxyType({})
    tasks_queued: int = 0
    tasks_running: int = 0
    tasks_done: int = 0
    tasks_failed: int = 0
    first_task_at: float = 0.0
    cluster: Cluster | None = None
    instances: tuple[Instance, ...] = ()
    metrics: MappingProxyType[int, MappingProxyType[str, float]] = MappingProxyType({})
    pool_started_at: float = 0.0
    task_latencies: tuple[float, ...] = ()
    inflight: MappingProxyType[str, _TaskEntry] = MappingProxyType({})
    task_fn_stats: MappingProxyType[str, tuple[float, ...]] = MappingProxyType({})
    task_fn_failed: MappingProxyType[str, int] = MappingProxyType({})
    ready_at: float = 0.0
    desired_nodes: int = 0
    pending_nodes: int = 0
    draining_nodes: int = 0
    reconciler_state: str = "watching"
    min_nodes: int | None = None
    max_nodes: int | None = None
    is_elastic: bool = False
    spec_accelerator_memory: str = ""
    tasks_per_node: MappingProxyType[int, int] = MappingProxyType({})
    ssh_user: str = ""
    ssh_key_path: str = ""
    bootstrap_spinners: MappingProxyType[int, _BootstrapTimeline] = MappingProxyType({})
    bootstrap_started: MappingProxyType[int, float] = MappingProxyType({})
    progress_lines: MappingProxyType[int, str] = MappingProxyType({})
    provision_error_shown: bool = False
