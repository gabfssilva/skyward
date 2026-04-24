from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from casty import ActorRef

from skyward.actors.messages import (
    ClusterReady,
    GetCurrentNodes,
    GetPoolSnapshot,
    HeadAddressKnown,
    NodeActivated,
    NodeBecameReady,
    NodeBecameUnready,
    NodeConnected,
    NodeExhausted,
    NodeLost,
    ReconciliationExhausted,
    RequestDrainNodes,
    RequestScaleDown,
    RequestScaleUp,
    SubmitBroadcast,
    SubmitTask,
)
from skyward.api.spec import Nodes

if TYPE_CHECKING:
    from skyward.core.model import Cluster, Instance, Offer
    from skyward.core.provider import ProviderConfig
    from skyward.core.spec import PoolSpec


@dataclass(frozen=True, slots=True)
class StartPool:
    spec: PoolSpec
    provider_config: ProviderConfig
    provider: Any
    offers: tuple[Offer, ...]
    reply_to: ActorRef[PoolStarted | ProvisionFailed]


@dataclass(frozen=True, slots=True)
class StopPool:
    reply_to: ActorRef[PoolStopped]


@dataclass(frozen=True, slots=True)
class PoolStarted:
    cluster_id: str
    instances: tuple[Any, ...]
    cluster: Cluster[Any]


@dataclass(frozen=True, slots=True)
class PoolStopped:
    pass


@dataclass(frozen=True, slots=True)
class ProvisionFailed:
    """Provisioning failed after exhausting all retry attempts."""

    reason: str


@dataclass(frozen=True, slots=True)
class InstancesProvisioned:
    cluster: Cluster
    instances: tuple[Instance, ...]


@dataclass(frozen=True, slots=True)
class RecoverPool:
    """Recover a pool from pre-existing instances (crash recovery).

    Skips prepare() and provision() -- goes directly to spawning
    node actors with already-provisioned instances.
    """
    spec: PoolSpec
    provider: Any
    cluster: Cluster[Any]
    instances: tuple[Instance, ...]
    reply_to: ActorRef[PoolStarted | ProvisionFailed]


@dataclass(frozen=True, slots=True)
class _ShutdownDone:
    pass


@dataclass(frozen=True, slots=True)
class Resize:
    """Reshape pool bounds and desired count in-flight.

    Accepted in the ``ready`` state only.  ``nodes`` is always a
    normalized ``Nodes`` instance — callers use ``Nodes.from_spec`` at
    the edge.
    """

    nodes: Nodes


type PoolMsg = (
    StartPool
    | StopPool
    | RecoverPool
    | PoolStarted
    | ProvisionFailed
    | HeadAddressKnown
    | NodeActivated
    | NodeBecameReady
    | NodeConnected
    | NodeBecameUnready
    | NodeLost
    | NodeExhausted
    | ReconciliationExhausted
    | SubmitTask
    | SubmitBroadcast
    | ClusterReady
    | InstancesProvisioned
    | _ShutdownDone
    | RequestScaleUp
    | RequestScaleDown
    | RequestDrainNodes
    | GetCurrentNodes
    | GetPoolSnapshot
    | Resize
)
