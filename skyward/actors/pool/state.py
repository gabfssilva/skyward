from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from casty import ActorRef, ClusterClient

from skyward.actors.messages import NodeBecameReady, NodeInstance

if TYPE_CHECKING:
    from skyward.actors.messages import ClusterId, NodeId
    from skyward.api.model import Offer
    from skyward.api.spec import PoolSpec


@dataclass(frozen=True, slots=True)
class PoolState:
    spec: PoolSpec
    provider: Any
    reply_to: ActorRef
    remaining_offers: tuple[Offer, ...] = ()
    cluster: Any = None
    cluster_id: ClusterId = ""
    instances: dict[NodeId, NodeInstance] = field(default_factory=dict)
    node_refs: dict[NodeId, ActorRef] = field(default_factory=dict)
    tm_ref: ActorRef | None = None
    head_addr: str | None = None
    client: ClusterClient | None = None
    ready_nodes: frozenset[int] = frozenset()
    reconciler_ref: ActorRef | None = None
    instance_map: dict[NodeId, str] | None = None
    attempt: int = 1
    early_ready: tuple[NodeBecameReady, ...] = ()
