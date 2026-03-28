from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from casty import ActorRef

from skyward.actors.pool.messages import PoolMsg
from skyward.actors.snapshot import PoolSnapshot

if TYPE_CHECKING:
    from skyward.core.model import Cluster, Instance, Offer
    from skyward.core.provider import ProviderConfig
    from skyward.core.spec import PoolSpec

type PoolPhase = str


@dataclass(frozen=True, slots=True)
class PoolInfo:
    name: str
    ref: ActorRef[PoolMsg]
    spec: PoolSpec
    phase: PoolPhase
    nodes_ready: int
    nodes_total: int


@dataclass(frozen=True, slots=True)
class SpawnPool:
    name: str
    spec: PoolSpec
    provider_config: ProviderConfig
    provider: Any
    offers: tuple[Offer, ...]
    provision_timeout: float
    reply_to: ActorRef[PoolSpawned | PoolSpawnFailed]


@dataclass(frozen=True, slots=True)
class PoolSpawned:
    name: str
    pool_ref: ActorRef[PoolMsg]
    cluster_id: str
    instances: tuple[Any, ...]
    cluster: Cluster[Any]


@dataclass(frozen=True, slots=True)
class PoolSpawnFailed:
    name: str
    reason: str


@dataclass(frozen=True, slots=True)
class PoolStateChanged:
    name: str
    phase: PoolPhase
    nodes_ready: int
    nodes_total: int


@dataclass(frozen=True, slots=True)
class StopSession:
    reply_to: ActorRef[SessionStopped]


@dataclass(frozen=True, slots=True)
class SessionStopped:
    pass


@dataclass(frozen=True, slots=True)
class GetSessionSnapshot:
    reply_to: ActorRef[SessionSnapshot]


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    pools: tuple[PoolSnapshot, ...]


@dataclass(frozen=True, slots=True)
class _SnapshotReady:
    snapshots: tuple[PoolSnapshot, ...]
    reply_to: ActorRef[SessionSnapshot]


@dataclass(frozen=True, slots=True)
class _PoolReady:
    name: str
    cluster_id: str
    instances: tuple[Any, ...]
    cluster: Any
    pool_ref: ActorRef[PoolMsg]


@dataclass(frozen=True, slots=True)
class _PoolFailed:
    name: str
    reason: str


@dataclass(frozen=True, slots=True)
class RecoverExistingPool:
    """Recover a pool from pre-existing instances (daemon crash recovery)."""
    name: str
    spec: PoolSpec
    provider: Any
    cluster: Cluster[Any]
    instances: tuple[Instance, ...]
    reply_to: ActorRef[PoolSpawned | PoolSpawnFailed]


type SessionMsg = (
    SpawnPool
    | RecoverExistingPool
    | StopSession
    | PoolStateChanged
    | GetSessionSnapshot
    | _SnapshotReady
    | _PoolReady
    | _PoolFailed
)
