"""The vocabulary of Skyward — every message, event, and type alias.

Two kinds of messages:
- System events: ClusterRequested, InstanceProvisioned, etc.
- Actor messages: StartPool, Provision, StopMonitor, etc.

Actor message types (PoolMsg, NodeMsg, MonitorMsg)
are the contracts — the type union IS the actor's public API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from casty import ActorRef

if TYPE_CHECKING:
    from skyward.core.model import Cluster, Instance
    from skyward.core.spec import PoolSpec
    from skyward.providers.provider import Provider

type RequestId = str
type ClusterId = str
type InstanceId = str
type NodeId = int
type ProviderName = Literal["aws", "gcp", "vastai", "verda", "runpod", "container"]


# =============================================================================
# Core Value Objects
# =============================================================================


@dataclass(frozen=True, slots=True)
class NodeInstance:
    """Instance bound to a node — infrastructure context + offer."""

    instance: Instance
    node: NodeId
    provider: ProviderName
    ssh_user: str
    ssh_key_path: str
    network_interface: str = ""


@dataclass
class InstanceRegistry:
    """Tracks active instances for monitoring."""

    _instances: dict[InstanceId, NodeInstance] = field(default_factory=dict)

    def register(self, info: NodeInstance) -> None:
        self._instances[info.instance.id] = info

    def unregister(self, instance_id: InstanceId) -> None:
        self._instances.pop(instance_id, None)

    @property
    def instances(self) -> list[NodeInstance]:
        return list(self._instances.values())

    @property
    def spot_instances(self) -> list[NodeInstance]:
        return [i for i in self._instances.values() if i.instance.spot]

    def get(self, instance_id: InstanceId) -> NodeInstance | None:
        return self._instances.get(instance_id)


# =============================================================================
# System Events — Requests (Commands)
# =============================================================================


@dataclass(frozen=True, slots=True)
class ClusterRequested:
    """Pool requests a new cluster from a provider."""

    request_id: RequestId
    provider: ProviderName
    spec: PoolSpec
    reply_to: ActorRef | None = None


@dataclass(frozen=True, slots=True)
class InstanceRequested:
    """Node requests an instance (new or replacement)."""

    request_id: RequestId
    provider: ProviderName
    cluster_id: ClusterId
    node_id: NodeId
    reply_to: ActorRef | None = None
    replacing: InstanceId | None = None


@dataclass(frozen=True, slots=True)
class ShutdownRequested:
    """Pool requests cluster shutdown."""

    cluster_id: ClusterId
    reply_to: ActorRef[ShutdownCompleted] | None = None


@dataclass(frozen=True, slots=True)
class ShutdownCompleted:
    """Provider confirms cluster shutdown is done."""

    cluster_id: ClusterId


@dataclass(frozen=True, slots=True)
class BootstrapRequested:
    """Request bootstrap on a running instance."""

    request_id: RequestId
    instance: NodeInstance
    cluster_id: ClusterId
    reply_to: ActorRef | None = None


# =============================================================================
# System Events — Facts (Immutable records of what happened)
# =============================================================================


@dataclass(frozen=True, slots=True)
class ClusterProvisioned:
    """Cluster infrastructure is ready."""

    request_id: RequestId
    cluster_id: ClusterId
    provider: ProviderName


@dataclass(frozen=True, slots=True)
class InstanceLaunched:
    """Provider launched an instance, waiting for running state."""

    request_id: RequestId
    cluster_id: ClusterId
    node_id: NodeId
    provider: ProviderName
    instance_id: str


@dataclass(frozen=True, slots=True)
class InstanceRunning:
    """Instance is running, ready for bootstrap."""

    request_id: RequestId
    cluster_id: ClusterId
    node_id: NodeId
    provider: ProviderName
    instance: Instance
    ssh_user: str = ""
    ssh_key_path: str = ""
    network_interface: str = ""


@dataclass(frozen=True, slots=True)
class InstanceProvisioned:
    """Instance was created (not yet bootstrapped)."""

    request_id: RequestId
    instance: NodeInstance


@dataclass(frozen=True, slots=True)
class InstanceBootstrapped:
    """Instance finished bootstrap, ready for work."""

    instance: NodeInstance


@dataclass(frozen=True, slots=True)
class InstancePreempted:
    """Instance was preempted (spot interruption)."""

    instance: NodeInstance
    reason: str


@dataclass(frozen=True, slots=True)
class InstanceReplaced:
    """Instance was successfully replaced after preemption."""

    request_id: RequestId
    old_id: InstanceId
    new: NodeInstance


@dataclass(frozen=True, slots=True)
class InstanceDestroyed:
    """Instance was terminated."""

    instance_id: InstanceId


@dataclass(frozen=True, slots=True)
class ClusterDestroyed:
    """Cluster was fully shut down."""

    cluster_id: ClusterId


@dataclass(frozen=True, slots=True)
class TaskStarted:
    """Task execution started on an instance."""

    task_id: str
    instance: NodeInstance
    function_name: str


@dataclass(frozen=True, slots=True)
class TaskCompleted:
    """Task execution completed."""

    task_id: str
    instance: NodeInstance
    duration: float
    success: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class Metric:
    """Metric value from an instance."""

    instance: NodeInstance
    name: str
    value: float
    timestamp: float


@dataclass(frozen=True, slots=True)
class Log:
    """Log line from an instance."""

    instance: NodeInstance
    line: str
    stream: Literal["stdout", "stderr"] = "stdout"


@dataclass(frozen=True, slots=True)
class BootstrapConsole:
    """Console output during bootstrap."""

    instance: NodeInstance
    content: str
    stream: Literal["stdout", "stderr"] = "stdout"


@dataclass(frozen=True, slots=True)
class BootstrapPhase:
    """Phase event during bootstrap."""

    instance: NodeInstance
    event: Literal["started", "completed", "failed"]
    phase: str
    elapsed: float | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class BootstrapCommand:
    """Command being executed during bootstrap phase."""

    instance: NodeInstance
    command: str


@dataclass(frozen=True, slots=True)
class BootstrapFailed:
    """Bootstrap failed on instance."""

    instance: NodeInstance
    phase: str
    error: str


@dataclass(frozen=True, slots=True)
class Error:
    """Something went wrong."""

    request_id: RequestId
    message: str
    fatal: bool = False


type Request = ClusterRequested | InstanceRequested | ShutdownRequested | BootstrapRequested

type Fact = (
    ClusterProvisioned
    | InstanceLaunched
    | InstanceRunning
    | InstanceProvisioned
    | InstanceBootstrapped
    | InstancePreempted
    | InstanceReplaced
    | InstanceDestroyed
    | ClusterReady
    | ClusterDestroyed
    | TaskStarted
    | TaskCompleted
    | Metric
    | Log
    | BootstrapConsole
    | BootstrapPhase
    | BootstrapCommand
    | BootstrapFailed
    | Error
)

type Event = Request | Fact


# =============================================================================
# Node Actor Messages (shared across actors)
# =============================================================================


@dataclass(frozen=True, slots=True)
class Preempted:
    reason: str = "preempted"


# =============================================================================
# Node Actor Messages
# =============================================================================


@dataclass(frozen=True, slots=True)
class Provision:
    cluster: Cluster[Any]
    provider: Provider[Any, Any]
    instance: Instance


@dataclass(frozen=True, slots=True)
class ExecuteOnNode:
    fn: Any
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    reply_to: ActorRef[Any]
    task_id: str = ""
    timeout: float = 600.0


@dataclass(frozen=True, slots=True)
class TaskResult:
    value: Any
    node_id: NodeId
    task_id: str = ""
    error: bool = False




# =============================================================================
# Pool Actor Messages
# =============================================================================


@dataclass(frozen=True, slots=True)
class NodeBecameReady:
    node_id: NodeId
    instance: NodeInstance
    local_port: int = 0
    private_ip: str = ""
    casty_port: int = 25520


@dataclass(frozen=True, slots=True)
class NodeActivated:
    node_id: NodeId
    node_ref: ActorRef[Any]
    slots: int


@dataclass(frozen=True, slots=True)
class NodeLost:
    node_id: NodeId
    reason: str


@dataclass(frozen=True, slots=True)
class HeadAddressKnown:
    head_addr: str
    casty_port: int
    num_nodes: int
    worker_concurrency: int
    worker_executor: str


@dataclass(frozen=True, slots=True)
class ClusterReady:
    cluster: Cluster




# =============================================================================
# TaskManager Actor Messages
# =============================================================================


@dataclass(frozen=True, slots=True)
class SubmitTask:
    fn: Any
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    reply_to: ActorRef[Any]
    task_id: str = field(default_factory=lambda: uuid4().hex[:8])
    timeout: float = 600.0


@dataclass(frozen=True, slots=True)
class SubmitBroadcast:
    fn: Any
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    reply_to: ActorRef[Any]
    task_id: str = field(default_factory=lambda: uuid4().hex[:8])
    timeout: float = 600.0


@dataclass(frozen=True, slots=True)
class NodeAvailable:
    node_id: NodeId
    node_ref: ActorRef[Any]
    slots: int


@dataclass(frozen=True, slots=True)
class NodeUnavailable:
    node_id: NodeId


@dataclass(frozen=True, slots=True)
class NodeSlots:
    ref: ActorRef[Any]
    total: int
    used: int


@dataclass(frozen=True, slots=True)
class TaskSubmitted:
    task_id: str
    node_id: NodeId


# ── Autoscaler ────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PressureReport:
    queued: int
    inflight: int
    total_capacity: int
    node_count: int



# ── Reconciler ────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class DesiredCountChanged:
    desired: int
    reason: str


@dataclass(frozen=True, slots=True)
class ReconcilerNodeLost:
    node_id: NodeId
    reason: str


@dataclass(frozen=True, slots=True)
class NodeJoined:
    node_id: NodeId


# ── Pool ↔ Reconciler ────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SpawnNodes:
    instances: tuple[Any, ...]
    cluster: Any
    start_node_id: int


@dataclass(frozen=True, slots=True)
class DrainNode:
    node_id: NodeId
    reply_to: ActorRef[DrainComplete]


@dataclass(frozen=True, slots=True)
class DrainComplete:
    node_id: NodeId
    instance_id: str


# ── Pool Query ────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class GetCurrentNodes:
    reply_to: ActorRef[CurrentNodeCount]


@dataclass(frozen=True, slots=True)
class CurrentNodeCount:
    count: int
    ready: int


@dataclass(frozen=True, slots=True)
class RegisterPressureObserver:
    observer: ActorRef[PressureReport]



# =============================================================================
# Provider Actor Messages (legacy — kept for old handler.py files)
# =============================================================================


@dataclass(frozen=True, slots=True)
class InstanceReady:
    """Internal: polling confirmed instance is running with IP."""

    instance_id: str
    node_id: int
    ip: str
    private_ip: str | None
    ssh_port: int
    spot: bool
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class BootstrapDone:
    """Internal: bootstrap monitor signals completion."""

    instance: NodeInstance
    success: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class _ProvisioningDone:
    state: Any


@dataclass(frozen=True, slots=True)
class _InstanceNowRunning:
    event: InstanceRunning


@dataclass(frozen=True, slots=True)
class _InstanceWaitFailed:
    instance_id: str
    node_id: int
    error: str


@dataclass(frozen=True, slots=True)
class _BootstrapScriptDone:
    instance_id: str


@dataclass(frozen=True, slots=True)
class _BootstrapScriptFailed:
    instance_id: str
    error: str


# =============================================================================
# Monitor Actor Messages
# =============================================================================


@dataclass(frozen=True, slots=True)
class StopMonitor:
    pass


# =============================================================================
# Helpers
# =============================================================================


def _bind_to_node(
    inst: Instance,
    node_id: NodeId,
    provider: ProviderName,
    cluster: Cluster,  # type: ignore[type-arg]
) -> NodeInstance:
    return NodeInstance(
        instance=inst,
        node=node_id,
        provider=provider,
        ssh_user=cluster.ssh_user,
        ssh_key_path=cluster.ssh_key_path,
    )
