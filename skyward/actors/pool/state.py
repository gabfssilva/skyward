from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from casty import ActorRef, ClusterClient

from skyward.actors.messages import NodeBecameReady, NodeInstance
from skyward.actors.snapshot import (
    BootstrapTimeline,
    NodeSnapshot,
    NodeStatus,
    PoolPhase,
    PoolSnapshot,
    ScalingSnapshot,
    TaskCounters,
)

if TYPE_CHECKING:
    from skyward.actors.messages import ClusterId, NodeId
    from skyward.core.model import Offer
    from skyward.core.spec import PoolSpec
    from skyward.infra.tls import CertificateAuthority


@dataclass(frozen=True, slots=True)
class PoolState:
    spec: PoolSpec
    provider: Any
    reply_to: ActorRef
    remaining_offers: tuple[Offer, ...] = ()
    cluster: Any = None
    cluster_id: ClusterId = ""
    instances: MappingProxyType[NodeId, NodeInstance] = MappingProxyType({})
    node_refs: MappingProxyType[NodeId, ActorRef] = MappingProxyType({})
    tm_ref: ActorRef | None = None
    head_addr: str | None = None
    client: ClusterClient | None = None
    clients: MappingProxyType[NodeId, ClusterClient] = MappingProxyType({})
    ready_nodes: frozenset[int] = frozenset()
    reconciler_ref: ActorRef | None = None
    instance_map: MappingProxyType[NodeId, str] = MappingProxyType({})
    attempt: int = 1
    early_ready: tuple[NodeBecameReady, ...] = ()
    ca: CertificateAuthority | None = None
    client_tls: Any | None = None
    phase: PoolPhase = PoolPhase.PROVISIONING
    node_statuses: MappingProxyType[str, NodeStatus] = MappingProxyType({})
    bootstrap_timelines: MappingProxyType[str, BootstrapTimeline] = MappingProxyType({})
    task_counters: TaskCounters = TaskCounters()
    scaling: ScalingSnapshot = ScalingSnapshot()
    pool_started_at: float = 0.0


def _derive_phase(s: PoolState) -> PoolPhase:
    """Derive pool phase from node statuses.

    Explicit phase (READY, WORKERS, STOPPING) takes priority — those are
    set by the pool actor's state machine. For earlier phases, derive from
    the minimum progress across all tracked nodes. Only advances past
    PROVISIONING when all expected nodes have reported in.
    """
    match s.phase:
        case PoolPhase.READY | PoolPhase.WORKERS | PoolPhase.STOPPING:
            return s.phase
    statuses = tuple(s.node_statuses.values())
    if not statuses or len(statuses) < s.spec.nodes.min:
        return PoolPhase.PROVISIONING
    min_status = min(statuses, key=lambda v: v.value)
    match min_status:
        case NodeStatus.READY:
            return PoolPhase.WORKERS
        case NodeStatus.BOOTSTRAPPING:
            return PoolPhase.BOOTSTRAP
        case NodeStatus.SSH:
            return PoolPhase.SSH
        case _:
            return PoolPhase.PROVISIONING


def build_pool_snapshot(s: PoolState, name: str) -> PoolSnapshot:
    nodes = tuple(
        NodeSnapshot(
            node_id=nid,
            instance_id=s.instances[nid].instance.id if nid in s.instances else "",
            status=s.node_statuses.get(
                s.instances[nid].instance.id if nid in s.instances else "",
                NodeStatus.WAITING,
            ),
            bootstrap=s.bootstrap_timelines.get(
                s.instances[nid].instance.id if nid in s.instances else "",
            ),
        )
        for nid in sorted(s.node_refs.keys())
    )
    return PoolSnapshot(
        name=name,
        phase=_derive_phase(s),
        nodes=nodes,
        tasks=s.task_counters,
        scaling=s.scaling,
        cluster=s.cluster,
        instances=tuple(s.cluster.instances) if s.cluster and s.cluster.instances else (),
        started_at=s.pool_started_at,
    )
