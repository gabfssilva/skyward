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
        phase=s.phase,
        nodes=nodes,
        tasks=s.task_counters,
        scaling=s.scaling,
        cluster=s.cluster,
        instances=tuple(s.cluster.instances) if s.cluster and s.cluster.instances else (),
        started_at=s.pool_started_at,
    )
