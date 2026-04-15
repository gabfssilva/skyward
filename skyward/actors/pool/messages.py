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
    RequestScaleDown,
    RequestScaleUp,
    SubmitBroadcast,
    SubmitTask,
)

if TYPE_CHECKING:
    from skyward.api.spec import Spec
    from skyward.core.model import Cluster, Instance, Offer
    from skyward.core.provider import ProviderConfig
    from skyward.core.spec import PoolSpec
    from skyward.server.host.domain import ComputeSpec


@dataclass(frozen=True, slots=True)
class StartPool:
    """Command the pool actor to provision a fresh cluster.

    ``spec`` is the resolved ``PoolSpec`` used for runtime decisions;
    ``compute_spec`` and ``chosen_spec`` are the authoritative user-
    facing projections that land in the Store so that recovery can
    rebuild the pool from persisted state.
    """

    spec: PoolSpec
    provider_config: ProviderConfig
    provider: Any
    offers: tuple[Offer, ...]
    compute_spec: ComputeSpec
    chosen_spec: Spec
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
    node actors with already-provisioned instances. ``compute_spec``
    and ``chosen_spec`` mirror the fields on :class:`StartPool` so
    that persisted state stays consistent across restarts.
    """
    spec: PoolSpec
    provider: Any
    cluster: Cluster[Any]
    instances: tuple[Instance, ...]
    compute_spec: ComputeSpec
    chosen_spec: Spec
    reply_to: ActorRef[PoolStarted | ProvisionFailed]


@dataclass(frozen=True, slots=True)
class _ShutdownDone:
    pass


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
    | GetCurrentNodes
    | GetPoolSnapshot
)
