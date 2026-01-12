"""
Events - the language of the system.

Two types:
- Requests (commands): what we want to happen
- Facts: what happened (immutable history)

Naming convention:
- *Requested: a component wants something
- *Provisioned/*Ready/*Destroyed: something happened
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .spec import PoolSpec

# =============================================================================
# Type Aliases
# =============================================================================

type RequestId = str
type ClusterId = str
type InstanceId = str
type NodeId = int
type ProviderName = Literal["aws", "digitalocean", "vastai", "verda"]


# =============================================================================
# Value Objects
# =============================================================================


@dataclass(frozen=True, slots=True)
class InstanceInfo:
    """Immutable snapshot of an instance."""

    id: InstanceId
    node: NodeId
    provider: ProviderName
    ip: str  # Public IP (for SSH from outside)
    private_ip: str = ""  # Private IP (for inter-node communication)
    network_interface: str = ""  # Network interface for NCCL (e.g., eth1 for overlay)
    spot: bool = False
    ssh_port: int = 22
    # Pricing info (populated by providers)
    hourly_rate: float = 0.0  # Actual rate for this instance (USD/hr)
    on_demand_rate: float = 0.0  # On-demand equivalent (for savings calc)
    billing_increment: int = 1  # Billing increment in minutes (1=AWS, per-second for VastAI)
    # Instance details
    instance_type: str = ""  # e.g., "p4d.24xlarge", "RTX 4090"
    gpu_count: int = 0  # Number of GPUs
    gpu_model: str = ""  # e.g., "A100", "H100"
    # Hardware specs (for Panel display)
    vcpus: int = 0  # vCPUs
    memory_gb: float = 0.0  # System RAM in GB
    gpu_vram_gb: int = 0  # VRAM per GPU in GB


# =============================================================================
# Requests (Commands) - what we want to happen
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
class ShutdownRequested:
    """Pool requests cluster shutdown."""

    cluster_id: ClusterId


@dataclass(frozen=True, slots=True)
class BootstrapRequested:
    """Request bootstrap on a running instance.

    Part of Event Pipeline - emitted by InstanceOrchestrator
    after InstanceRunning is received.
    """

    request_id: RequestId
    instance: InstanceInfo
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


# -----------------------------------------------------------------------------
# Event Pipeline - Intermediate events for decoupled instance lifecycle
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InstanceLaunched:
    """Provider launched an instance, waiting for running state.

    Part of Event Pipeline - emitted by provider handler
    after API call to create instance succeeds.
    """

    request_id: RequestId
    cluster_id: ClusterId
    node_id: NodeId
    provider: ProviderName
    instance_id: str  # Raw provider instance ID


@dataclass(frozen=True, slots=True)
class InstanceRunning:
    """Instance is running, ready for bootstrap.

    Part of Event Pipeline - emitted by provider handler
    after polling confirms instance is running with IP.
    """

    request_id: RequestId
    cluster_id: ClusterId
    node_id: NodeId
    provider: ProviderName
    instance_id: str
    ip: str
    private_ip: str | None
    ssh_port: int
    spot: bool
    network_interface: str = ""  # Optional NCCL interface
    # Pricing info (populated by providers)
    hourly_rate: float = 0.0  # Actual rate for this instance (USD/hr)
    on_demand_rate: float = 0.0  # On-demand equivalent (for savings calc)
    billing_increment: int = 1  # Billing increment in minutes
    # Instance details
    instance_type: str = ""  # e.g., "p4d.24xlarge"
    gpu_count: int = 0  # Number of GPUs
    gpu_model: str = ""  # e.g., "A100"
    # Hardware specs (for Panel display)
    vcpus: int = 0  # vCPUs
    memory_gb: float = 0.0  # System RAM in GB
    gpu_vram_gb: int = 0  # VRAM per GPU in GB


# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InstanceProvisioned:
    """Instance was created (not yet bootstrapped)."""

    request_id: RequestId
    instance: InstanceInfo


@dataclass(frozen=True, slots=True)
class InstanceBootstrapped:
    """Instance finished bootstrap, ready for work."""

    instance: InstanceInfo


@dataclass(frozen=True, slots=True)
class InstancePreempted:
    """Instance was preempted (spot interruption)."""

    instance: InstanceInfo
    reason: str  # "spot-interruption", "maintenance", "outbid"


@dataclass(frozen=True, slots=True)
class InstanceReplaced:
    """Instance was successfully replaced after preemption."""

    request_id: RequestId
    old_id: InstanceId
    new: InstanceInfo


@dataclass(frozen=True, slots=True)
class InstanceDestroyed:
    """Instance was terminated."""

    instance_id: InstanceId


@dataclass(frozen=True, slots=True)
class NodeReady:
    """Node signals its instance is ready."""

    node_id: NodeId
    instance: InstanceInfo


@dataclass(frozen=True, slots=True)
class ClusterReady:
    """All nodes are ready - cluster is operational."""

    cluster_id: ClusterId
    nodes: tuple[InstanceInfo, ...]


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
    instance: InstanceInfo
    function_name: str


@dataclass(frozen=True, slots=True)
class TaskCompleted:
    """Task execution completed."""

    task_id: str
    instance: InstanceInfo
    duration: float
    success: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class Metric:
    """Metric value from an instance."""

    instance: InstanceInfo
    name: str
    value: float
    timestamp: float


@dataclass(frozen=True, slots=True)
class Log:
    """Log line from an instance."""

    instance: InstanceInfo
    line: str
    stream: Literal["stdout", "stderr"] = "stdout"


# =============================================================================
# Bootstrap Streaming Events (from JSONL)
# =============================================================================


@dataclass(frozen=True, slots=True)
class BootstrapConsole:
    """Console output during bootstrap (from JSONL stream).

    These events are parsed from the events.jsonl file on the instance
    and represent real-time stdout/stderr output during bootstrap phases.
    """

    instance: InstanceInfo
    content: str
    stream: Literal["stdout", "stderr"] = "stdout"


@dataclass(frozen=True, slots=True)
class BootstrapPhase:
    """Phase event during bootstrap (from JSONL stream).

    These events mark the start, completion, or failure of bootstrap phases
    like "apt", "pip", "uv", "skyward", etc.
    """

    instance: InstanceInfo
    event: Literal["started", "completed", "failed"]
    phase: str  # "apt", "pip", "uv", "skyward", "bootstrap"
    elapsed: float | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class BootstrapCommand:
    """Command being executed during bootstrap phase (from JSONL stream).

    Emitted after BootstrapPhase(event="started") to show the actual
    command being executed in that phase.
    """

    instance: InstanceInfo
    command: str


@dataclass(frozen=True, slots=True)
class BootstrapFailed:
    """Bootstrap failed on instance.

    Emitted when bootstrap fails, with error details.
    """

    instance: InstanceInfo
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
# Type Unions
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
    | NodeReady
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
# Exports
# =============================================================================

__all__ = [
    # Type aliases
    "RequestId",
    "ClusterId",
    "InstanceId",
    "NodeId",
    "ProviderName",
    # Value objects
    "InstanceInfo",
    # Requests
    "ClusterRequested",
    "InstanceRequested",
    "ShutdownRequested",
    "BootstrapRequested",
    # Facts
    "ClusterProvisioned",
    # Event Pipeline
    "InstanceLaunched",
    "InstanceRunning",
    # Instance lifecycle
    "InstanceProvisioned",
    "InstanceBootstrapped",
    "InstancePreempted",
    "InstanceReplaced",
    "InstanceDestroyed",
    "NodeReady",
    "ClusterReady",
    "ClusterDestroyed",
    # Execution
    "TaskStarted",
    "TaskCompleted",
    "Metric",
    "Log",
    # Bootstrap streaming
    "BootstrapConsole",
    "BootstrapPhase",
    "BootstrapCommand",
    "BootstrapFailed",
    # Error
    "Error",
    # Unions
    "Request",
    "Fact",
    "Event",
]
