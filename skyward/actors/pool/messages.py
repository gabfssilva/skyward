from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from casty import ActorRef

from skyward.actors.messages import (
    ClusterReady,
    DrainNode,
    GetCurrentNodes,
    HeadAddressKnown,
    NodeBecameReady,
    NodeLost,
    SpawnNodes,
    SubmitBroadcast,
    SubmitTask,
)

if TYPE_CHECKING:
    from skyward.api.model import Cluster, Instance, Offer
    from skyward.api.provider import ProviderConfig
    from skyward.api.spec import PoolSpec


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
class _ShutdownDone:
    pass


type PoolMsg = (
    StartPool
    | StopPool
    | PoolStarted
    | ProvisionFailed
    | HeadAddressKnown
    | NodeBecameReady
    | NodeLost
    | SubmitTask
    | SubmitBroadcast
    | ClusterReady
    | InstancesProvisioned
    | _ShutdownDone
    | SpawnNodes
    | DrainNode
    | GetCurrentNodes
)
