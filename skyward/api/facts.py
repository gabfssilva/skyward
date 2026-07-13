"""Domain facts — the record of what happened on a cluster.

Pure data: the identifier aliases, the instance binding (``NodeInstance``)
that every fact carries, and the facts themselves — provisioning,
bootstrap, task, metric, and log records streamed off the nodes.  The
``Fact`` union is the closed vocabulary the control plane speaks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from skyward.api.model import Cluster, Instance

__all__ = [
    "BootstrapCommand",
    "BootstrapFailed",
    "BootstrapPhase",
    "ClusterDestroyed",
    "ClusterId",
    "ClusterProvisioned",
    "ClusterReady",
    "ConsoleOutput",
    "Error",
    "Event",
    "Fact",
    "InstanceBootstrapped",
    "InstanceDestroyed",
    "InstanceId",
    "InstanceLaunched",
    "InstancePreempted",
    "InstanceProvisioned",
    "InstanceRegistry",
    "InstanceReplaced",
    "InstanceRunning",
    "Log",
    "Metric",
    "NodeId",
    "NodeInstance",
    "ProviderName",
    "RequestId",
    "TaskCompleted",
    "TaskStarted",
]

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
    ssh_password: str | None = None
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
# Facts — immutable records of what happened
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
    ssh_password: str | None = None
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
class ClusterReady:
    cluster: Cluster


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
    overwrite: bool = False


@dataclass(frozen=True, slots=True)
class ConsoleOutput:
    """Console output from a remote instance."""

    instance: NodeInstance
    content: str
    stream: Literal["stdout", "stderr"] = "stdout"
    overwrite: bool = False


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
    | ConsoleOutput
    | BootstrapPhase
    | BootstrapCommand
    | BootstrapFailed
    | Error
)

type Event = Fact
