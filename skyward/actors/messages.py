"""The vocabulary of Skyward — every message, event, and type alias.

Two kinds of messages:
- System events: ClusterRequested, InstanceProvisioned, etc.
- Actor messages: StartPool, Provision, StopMonitor, etc.

Actor message types (PoolMsg, NodeMsg, ProviderMsg, MonitorMsg)
are the contracts — the type union IS the actor's public API.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from casty import ActorRef

if TYPE_CHECKING:
    from skyward.api.spec import PoolSpec

# =============================================================================
# Type Aliases
# =============================================================================

type RequestId = str
type ClusterId = str
type InstanceId = str
type NodeId = int
type ProviderName = Literal["aws", "vastai", "verda", "runpod"]


# =============================================================================
# Value Objects
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
    hourly_rate: float = 0.0
    on_demand_rate: float = 0.0
    billing_increment: int = 1
    instance_type: str = ""
    gpu_count: int = 0
    gpu_model: str = ""
    vcpus: int = 0
    memory_gb: float = 0.0
    gpu_vram_gb: int = 0
    region: str = ""


# =============================================================================
# Requests (Commands)
# =============================================================================


@dataclass(frozen=True, slots=True)
class ClusterRequested:
    """Pool requests a new cluster from a provider."""

    request_id: RequestId
    provider: ProviderName
    spec: PoolSpec


@dataclass(frozen=True, slots=True)
class InstanceRequested:
    """Node requests an instance (new or replacement)."""

    request_id: RequestId
    provider: ProviderName
    cluster_id: ClusterId
    node_id: NodeId
    replacing: InstanceId | None = None


@dataclass(frozen=True, slots=True)
class ShutdownCompleted:
    """Provider confirms cluster shutdown is done."""

    cluster_id: ClusterId


@dataclass(frozen=True, slots=True)
class ShutdownRequested:
    """Pool requests cluster shutdown."""

    cluster_id: ClusterId
    reply_to: ActorRef[ShutdownCompleted] | None = None


@dataclass(frozen=True, slots=True)
class BootstrapRequested:
    """Request bootstrap on a running instance."""

    request_id: RequestId
    instance: InstanceMetadata
    cluster_id: ClusterId


# =============================================================================
# Facts - what happened (immutable)
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
    vcpus: int = 0
    memory_gb: float = 0.0
    gpu_vram_gb: int = 0
    region: str = ""


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
class ClusterReady:
    """All nodes are ready - cluster is operational."""

    cluster_id: ClusterId
    nodes: tuple[InstanceMetadata, ...]


@dataclass(frozen=True, slots=True)
class ClusterDestroyed:
    """Cluster was fully shut down."""

    cluster_id: ClusterId


# =============================================================================
# Execution Events
# =============================================================================


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


# =============================================================================
# Bootstrap Streaming Events (from JSONL)
# =============================================================================


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


# =============================================================================
# Error Events
# =============================================================================


@dataclass(frozen=True, slots=True)
class Error:
    """Something went wrong."""

    request_id: RequestId
    message: str
    fatal: bool = False


# =============================================================================
# Type Unions (system events)
# =============================================================================

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
# Instance Registry
# =============================================================================


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
# Node Messages
# =============================================================================


@dataclass(frozen=True, slots=True)
class Provision:
    cluster_id: ClusterId
    provider: ProviderName


@dataclass(frozen=True, slots=True)
class Replace:
    old_instance_id: InstanceId
    reason: str


type NodeMsg = (
    Provision
    | InstanceProvisioned
    | InstanceBootstrapped
    | InstancePreempted
)


# =============================================================================
# Pool Messages
# =============================================================================


@dataclass(frozen=True, slots=True)
class PoolStarted:
    cluster_id: ClusterId
    instances: tuple[InstanceMetadata, ...]


@dataclass(frozen=True, slots=True)
class PoolStopped:
    pass


@dataclass(frozen=True, slots=True)
class ExecuteResult:
    value: Any
    node_id: int


@dataclass(frozen=True, slots=True)
class BroadcastResult:
    values: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class StartPool:
    spec: PoolSpec
    provider_config: Any
    provider_ref: ActorRef[ProviderMsg]
    reply_to: ActorRef[PoolStarted]


@dataclass(frozen=True, slots=True)
class StopPool:
    reply_to: ActorRef[PoolStopped]


@dataclass(frozen=True, slots=True)
class ExecuteTask:
    fn: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    node: int | None
    reply_to: ActorRef[ExecuteResult]


@dataclass(frozen=True, slots=True)
class BroadcastTask:
    fn: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    reply_to: ActorRef[BroadcastResult]


@dataclass(frozen=True, slots=True)
class NodeBecameReady:
    node_id: NodeId
    instance: InstanceMetadata


type PoolMsg = (
    StartPool
    | StopPool
    | ExecuteTask
    | BroadcastTask
    | ClusterProvisioned
    | InstanceRunning
    | InstanceProvisioned
    | InstanceBootstrapped
    | InstancePreempted
    | NodeBecameReady
    | ShutdownCompleted
)


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
    )


# =============================================================================
# Provider Messages
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


@dataclass(frozen=True, slots=True)
class _LocalInstallDone:
    instance: InstanceMetadata


@dataclass(frozen=True, slots=True)
class _LocalInstallFailed:
    instance: InstanceMetadata
    error: str


type ProviderMsg = (
    ClusterRequested
    | InstanceRequested
    | BootstrapRequested
    | ShutdownRequested
    | InstanceReady
    | BootstrapDone
    | _ProvisioningDone
    | _InstanceNowRunning
    | _InstanceWaitFailed
    | _BootstrapScriptDone
    | _BootstrapScriptFailed
    | _LocalInstallDone
    | _LocalInstallFailed
)


# =============================================================================
# Monitor Messages
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
