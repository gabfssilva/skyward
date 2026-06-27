from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from casty import ActorRef

from skyward.actors.messages import (
    ClusterReady,
    FileOpOnNodes,
    FileOpReplies,
    GetCurrentNodes,
    GetPoolSnapshot,
    HeadAddressKnown,
    NodeActivated,
    NodeBecameReady,
    NodeBecameUnready,
    NodeConnected,
    NodeExhausted,
    NodeFileResult,
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

    Skips prepare() and provision() -- spawns node actors that ``Adopt``
    already-provisioned, bootstrapped, worker-running instances. ``node_ids``
    carries the persisted ranks so head (rank 0) is restored to its original
    node; ``()`` falls back to enumeration.
    """
    spec: PoolSpec
    provider: Any
    cluster: Cluster[Any]
    instances: tuple[Instance, ...]
    reply_to: ActorRef[PoolStarted | ProvisionFailed]
    node_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class _ShutdownDone:
    error: str | None = None


@dataclass(frozen=True, slots=True)
class Resize:
    """Reshape pool bounds and desired count in-flight.

    Accepted in the ``ready`` state only.  ``nodes`` is always a
    normalized ``Nodes`` instance — callers use ``Nodes.from_spec`` at
    the edge.
    """

    nodes: Nodes


@dataclass(frozen=True, slots=True)
class _FileOpGathered:
    results: tuple[NodeFileResult, ...]
    reply_to: ActorRef[FileOpReplies]


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
    | FileOpOnNodes
    | _FileOpGathered
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
