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
    from skyward.api.model import Cluster, Instance
    from skyward.api.provider import ProviderConfig
    from skyward.api.spec import PoolSpec

type RequestId = str
type ClusterId = str
type InstanceId = str
type NodeId = int
type ProviderName = Literal["aws", "vastai", "verda", "runpod"]


# =============================================================================
# Core Value Objects
# =============================================================================


@dataclass(frozen=True, slots=True)
class InstanceMetadata:
    """Immutable snapshot of an instance's infrastructure metadata."""

    id: InstanceId
    node: NodeId
    provider: ProviderName
    ip: str
    private_ip: str = ""
    network_interface: str = ""
    spot: bool = False
    ssh_port: int = 22
    ssh_user: str = ""
    ssh_key_path: str = ""
    hourly_rate: float = 0.0
    on_demand_rate: float = 0.0
    billing_increment: int = 1
    instance_type: str = ""
    gpu_count: int = 0
    gpu_model: str = ""
    vcpus: float = 0
    memory_gb: float = 0.0
    gpu_vram_gb: int = 0
    region: str = ""


@dataclass
class InstanceRegistry:
    """Tracks active instances for monitoring."""

    _instances: dict[InstanceId, InstanceMetadata] = field(default_factory=dict)

    def register(self, info: InstanceMetadata) -> None:
        self._instances[info.id] = info

    def unregister(self, instance_id: InstanceId) -> None:
        self._instances.pop(instance_id, None)

    @property
    def instances(self) -> list[InstanceMetadata]:
        return list(self._instances.values())

    @property
    def spot_instances(self) -> list[InstanceMetadata]:
        return [i for i in self._instances.values() if i.spot]

    def get(self, instance_id: InstanceId) -> InstanceMetadata | None:
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
    instance: InstanceMetadata
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
    instance_id: str
    ip: str
    private_ip: str | None
    ssh_port: int
    spot: bool
    network_interface: str = ""
    hourly_rate: float = 0.0
    on_demand_rate: float = 0.0
    billing_increment: int = 1
    instance_type: str = ""
    gpu_count: int = 0
    gpu_model: str = ""
    vcpus: float = 0
    memory_gb: float = 0.0
    gpu_vram_gb: int = 0
    region: str = ""
    ssh_user: str = ""
    ssh_key_path: str = ""


@dataclass(frozen=True, slots=True)
class InstanceProvisioned:
    """Instance was created (not yet bootstrapped)."""

    request_id: RequestId
    instance: InstanceMetadata


@dataclass(frozen=True, slots=True)
class InstanceBootstrapped:
    """Instance finished bootstrap, ready for work."""

    instance: InstanceMetadata


@dataclass(frozen=True, slots=True)
class InstancePreempted:
    """Instance was preempted (spot interruption)."""

    instance: InstanceMetadata
    reason: str


@dataclass(frozen=True, slots=True)
class InstanceReplaced:
    """Instance was successfully replaced after preemption."""

    request_id: RequestId
    old_id: InstanceId
    new: InstanceMetadata


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
    instance: InstanceMetadata
    function_name: str


@dataclass(frozen=True, slots=True)
class TaskCompleted:
    """Task execution completed."""

    task_id: str
    instance: InstanceMetadata
    duration: float
    success: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class Metric:
    """Metric value from an instance."""

    instance: InstanceMetadata
    name: str
    value: float
    timestamp: float


@dataclass(frozen=True, slots=True)
class Log:
    """Log line from an instance."""

    instance: InstanceMetadata
    line: str
    stream: Literal["stdout", "stderr"] = "stdout"


@dataclass(frozen=True, slots=True)
class BootstrapConsole:
    """Console output during bootstrap."""

    instance: InstanceMetadata
    content: str
    stream: Literal["stdout", "stderr"] = "stdout"


@dataclass(frozen=True, slots=True)
class BootstrapPhase:
    """Phase event during bootstrap."""

    instance: InstanceMetadata
    event: Literal["started", "completed", "failed"]
    phase: str
    elapsed: float | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class BootstrapCommand:
    """Command being executed during bootstrap phase."""

    instance: InstanceMetadata
    command: str


@dataclass(frozen=True, slots=True)
class BootstrapFailed:
    """Bootstrap failed on instance."""

    instance: InstanceMetadata
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
# Instance Actor Messages (internal to instance lifecycle)
# =============================================================================


@dataclass(frozen=True, slots=True)
class Running:
    ip: str


@dataclass(frozen=True, slots=True)
class Bootstrapping:
    phase: str
    status: str
    elapsed: float | None = None


@dataclass(frozen=True, slots=True)
class Bootstrapped:
    pass


@dataclass(frozen=True, slots=True)
class Preempted:
    reason: str = "preempted"


@dataclass(frozen=True, slots=True)
class Execute:
    fn_bytes: bytes
    reply_to: ActorRef[Any]
    task_id: str = ""


@dataclass(frozen=True, slots=True)
class _PollTick:
    pass


@dataclass(frozen=True, slots=True)
class _PollResult:
    instance: Any | None = None


@dataclass(frozen=True, slots=True)
class _LocalInstallDone:
    instance: InstanceMetadata


@dataclass(frozen=True, slots=True)
class _LocalInstallFailed:
    instance: InstanceMetadata
    error: str


@dataclass(frozen=True, slots=True)
class _UserCodeSyncDone:
    instance: InstanceMetadata


@dataclass(frozen=True, slots=True)
class _UserCodeSyncFailed:
    instance: InstanceMetadata
    error: str


@dataclass(frozen=True, slots=True)
class _PostBootstrapFailed:
    error: str


@dataclass(frozen=True, slots=True)
class _Connected:
    transport: Any
    listener: Any


@dataclass(frozen=True, slots=True)
class _ConnectionFailed:
    error: str


@dataclass(frozen=True, slots=True)
class _WorkerStarted:
    client: Any
    worker_ref: Any


@dataclass(frozen=True, slots=True)
class _WorkerFailed:
    error: str


@dataclass(frozen=True, slots=True)
class _BootstrapUploaded:
    pass


@dataclass(frozen=True, slots=True)
class _BootstrapUploadFailed:
    error: str


@dataclass(frozen=True, slots=True)
class _CorrelatedTaskResult:
    task_id: str
    value: Any
    node_id: int
    error: bool = False


type InstanceMsg = (
    Running | Bootstrapping | Bootstrapped | BootstrapDone
    | Log | Metric | Preempted | Execute | HeadAddressKnown
    | _PollTick | _PollResult
    | _LocalInstallDone | _LocalInstallFailed
    | _UserCodeSyncDone | _UserCodeSyncFailed
    | _PostBootstrapFailed
    | _Connected | _ConnectionFailed
    | _WorkerStarted | _WorkerFailed
    | _BootstrapUploaded | _BootstrapUploadFailed
    | _CorrelatedTaskResult
)


# =============================================================================
# Node Actor Messages
# =============================================================================


@dataclass(frozen=True, slots=True)
class Provision:
    cluster: Any
    provider: Any
    instance: Any


@dataclass(frozen=True, slots=True)
class InstanceBecameReady:
    instance_id: InstanceId
    ip: str
    metadata: InstanceMetadata | None = None


@dataclass(frozen=True, slots=True)
class InstanceDied:
    instance_id: InstanceId
    reason: str


@dataclass(frozen=True, slots=True)
class ExecuteOnNode:
    fn_bytes: bytes
    reply_to: ActorRef[Any]
    task_id: str = ""


@dataclass(frozen=True, slots=True)
class TaskResult:
    value: Any
    node_id: NodeId
    task_id: str = ""


type NodeMsg = (
    Provision
    | InstanceBecameReady
    | InstanceDied
    | ExecuteOnNode
    | TaskResult
    | HeadAddressKnown
)


# =============================================================================
# Pool Actor Messages
# =============================================================================


@dataclass(frozen=True, slots=True)
class NodeBecameReady:
    node_id: NodeId
    instance: InstanceMetadata


@dataclass(frozen=True, slots=True)
class NodeLost:
    node_id: NodeId
    reason: str


@dataclass(frozen=True, slots=True)
class PoolStarted:
    cluster_id: ClusterId
    instances: tuple[InstanceMetadata, ...]


@dataclass(frozen=True, slots=True)
class PoolStopped:
    pass


@dataclass(frozen=True, slots=True)
class StartPool:
    spec: PoolSpec
    provider_config: ProviderConfig
    provider: Any
    reply_to: ActorRef[PoolStarted]


@dataclass(frozen=True, slots=True)
class StopPool:
    reply_to: ActorRef[PoolStopped]


@dataclass(frozen=True, slots=True)
class HeadAddressKnown:
    head_addr: str
    casty_port: int
    num_nodes: int
    concurrency: int


@dataclass(frozen=True, slots=True)
class ClusterReady:
    cluster: Cluster


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
    | HeadAddressKnown
    | NodeBecameReady
    | NodeLost
    | SubmitTask
    | SubmitBroadcast
    | ClusterReady
    | InstancesProvisioned
    | _ShutdownDone
)


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


@dataclass(frozen=True, slots=True)
class SubmitBroadcast:
    fn: Any
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    reply_to: ActorRef[Any]
    task_id: str = field(default_factory=lambda: uuid4().hex[:8])


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


type TaskManagerMsg = (
    NodeAvailable | NodeUnavailable | TaskResult
    | SubmitTask | SubmitBroadcast | TaskSubmitted
)


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

    instance: InstanceMetadata
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


type ProviderMsg = (
    ClusterRequested
    | InstanceRequested
    | BootstrapRequested
    | ShutdownRequested
    | ShutdownCompleted
    | InstanceReady
    | BootstrapDone
    | _ProvisioningDone
    | _InstanceNowRunning
    | _InstanceWaitFailed
    | _BootstrapScriptDone
    | _BootstrapScriptFailed
    | _LocalInstallDone
    | _LocalInstallFailed
    | _UserCodeSyncDone
    | _UserCodeSyncFailed
)


# =============================================================================
# Monitor Actor Messages
# =============================================================================


@dataclass(frozen=True, slots=True)
class StopMonitor:
    pass


@dataclass(frozen=True, slots=True)
class _StreamedEvent:
    event: Event
    lines_read: int = 0


@dataclass(frozen=True, slots=True)
class _StreamEnded:
    error: str | None = None


type MonitorMsg = StopMonitor | _StreamedEvent | _StreamEnded


# =============================================================================
# Helpers
# =============================================================================


def _to_metadata(ev: InstanceRunning) -> InstanceMetadata:
    return InstanceMetadata(
        id=ev.instance_id,
        node=ev.node_id,
        provider=ev.provider,
        ip=ev.ip,
        private_ip=ev.private_ip or "",
        network_interface=ev.network_interface,
        spot=ev.spot,
        ssh_port=ev.ssh_port,
        hourly_rate=ev.hourly_rate,
        on_demand_rate=ev.on_demand_rate,
        billing_increment=ev.billing_increment,
        instance_type=ev.instance_type,
        gpu_count=ev.gpu_count,
        gpu_model=ev.gpu_model,
        vcpus=ev.vcpus,
        memory_gb=ev.memory_gb,
        gpu_vram_gb=ev.gpu_vram_gb,
        region=ev.region,
        ssh_user=ev.ssh_user,
        ssh_key_path=ev.ssh_key_path,
    )


def _instance_to_metadata(
    inst: Instance,
    node_id: NodeId,
    provider: ProviderName,
    cluster: Cluster,  # type: ignore[type-arg]
) -> InstanceMetadata:
    return InstanceMetadata(
        id=inst.id,
        node=node_id,
        provider=provider,
        ip=inst.ip or "",
        private_ip=inst.private_ip or "",
        spot=inst.spot,
        ssh_port=inst.ssh_port,
        ssh_user=cluster.ssh_user,
        ssh_key_path=cluster.ssh_key_path,
        hourly_rate=inst.hourly_rate,
        on_demand_rate=inst.on_demand_rate,
        billing_increment=inst.billing_increment,
        instance_type=inst.instance_type,
        gpu_count=inst.gpu_count,
        gpu_model=inst.gpu_model,
        vcpus=inst.vcpus,
        memory_gb=inst.memory_gb,
        gpu_vram_gb=inst.gpu_vram_gb,
        region=inst.region,
    )
