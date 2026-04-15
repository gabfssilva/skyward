from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from casty import ActorRef

from skyward.actors.pool.messages import PoolMsg

if TYPE_CHECKING:
    from skyward.api.spec import Spec
    from skyward.core.model import Cluster, Instance, Offer
    from skyward.core.provider import ProviderConfig
    from skyward.core.spec import PoolSpec
    from skyward.server.host.domain import ComputeSpec


@dataclass(frozen=True, slots=True)
class PoolInfo:
    name: str
    ref: ActorRef[PoolMsg]


@dataclass(frozen=True, slots=True)
class CreatePool:
    """Spawn a pool actor as a session child and return its ref immediately."""

    name: str
    reply_to: ActorRef[PoolCreated]


@dataclass(frozen=True, slots=True)
class PoolCreated:
    pool_ref: ActorRef[PoolMsg]


@dataclass(frozen=True, slots=True)
class SpawnPool:
    name: str
    spec: PoolSpec
    provider_config: ProviderConfig
    provider: Any
    offers: tuple[Offer, ...]
    provision_timeout: float
    compute_spec: ComputeSpec
    chosen_spec: Spec
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
class StopSession:
    reply_to: ActorRef[SessionStopped]


@dataclass(frozen=True, slots=True)
class SessionStopped:
    pass


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
    compute_spec: ComputeSpec
    chosen_spec: Spec
    reply_to: ActorRef[PoolSpawned | PoolSpawnFailed]


type SessionMsg = (
    CreatePool
    | SpawnPool
    | RecoverExistingPool
    | StopSession
    | _PoolReady
    | _PoolFailed
)
