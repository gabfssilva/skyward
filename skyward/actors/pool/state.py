from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from casty import ActorRef, ClusterClient

from skyward.actors.messages import (
    BootstrapCommand,
    BootstrapPhase,
    ConsoleOutput,
    NodeInstance,
)
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
    from skyward.actors.messages import (
        ClusterId,
        HeadAddressKnown,
        NodeBecameReady,
        NodeId,
    )
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
    dead_nodes: frozenset[int] = frozenset()
    reconciler_ref: ActorRef | None = None
    autoscaler_ref: ActorRef | None = None
    instance_map: MappingProxyType[NodeId, str] = MappingProxyType({})
    attempt: int = 1
    spawn_gen: int = 0
    ca: CertificateAuthority | None = None
    client_tls: Any | None = None
    phase: PoolPhase = PoolPhase.PROVISIONING
    node_statuses: MappingProxyType[str, NodeStatus] = MappingProxyType({})
    bootstrap_timelines: MappingProxyType[str, BootstrapTimeline] = MappingProxyType({})
    task_counters: TaskCounters = TaskCounters()
    scaling: ScalingSnapshot = ScalingSnapshot()
    pool_started_at: float = 0.0
    buffered_events: tuple[NodeBecameReady | HeadAddressKnown, ...] = ()
    node_transports: MappingProxyType[NodeId, ActorRef] = MappingProxyType({})
    proxy_refs: tuple[ActorRef, ...] = ()


def apply_stream_event(
    timelines: MappingProxyType[str, BootstrapTimeline],
    iid: str,
    msg: BootstrapPhase | BootstrapCommand | ConsoleOutput,
) -> MappingProxyType[str, BootstrapTimeline]:
    """Fold a monitor stream message into the per-instance bootstrap timeline."""
    tl = timelines.get(iid)
    match msg:
        case BootstrapPhase(event="started", phase=p) if p != "bootstrap":
            if tl is None:
                new = BootstrapTimeline(
                    phases=(p,), completed=frozenset(), active=p, output="",
                )
            else:
                completed = tl.completed | {tl.active} if tl.active else tl.completed
                phases = tl.phases if p in tl.phases else (*tl.phases, p)
                new = BootstrapTimeline(
                    phases=phases, completed=completed, active=p, output="",
                )
        case BootstrapPhase(event="completed", phase=p) if tl is not None and p != "bootstrap":
            new = replace(tl, completed=tl.completed | {p})
        case BootstrapCommand(command=cmd) if tl is not None:
            new = replace(tl, output=cmd[:80])
        case ConsoleOutput(content=c) if (
            tl is not None and (stripped := c.strip()) and not stripped.startswith("#")
        ):
            new = replace(tl, output=stripped)
        case _:
            return timelines
    return MappingProxyType({**timelines, iid: new})


def _derive_phase(s: PoolState) -> PoolPhase:
    """Derive pool phase from node statuses.

    Explicit phase (READY, WORKERS, STOPPED) takes priority — those are
    set by the pool actor's state machine. For earlier phases, derive from
    the minimum progress across all tracked nodes. Only advances past
    PROVISIONING when all expected nodes have reported in.
    """
    match s.phase:
        case PoolPhase.READY | PoolPhase.WORKERS | PoolPhase.STOPPED:
            return s.phase
    statuses = tuple(s.node_statuses.values())
    if not statuses or len(statuses) < s.spec.nodes.desired:
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
        instances=tuple(
            {
                **{i.id: i for i in (s.cluster.instances if s.cluster else ())},
                **{ni.instance.id: ni.instance for ni in s.instances.values()},
            }.values()
        ),
        started_at=s.pool_started_at,
    )
