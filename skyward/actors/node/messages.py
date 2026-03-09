from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from casty import ActorRef, Terminated

from skyward.actors.messages import (
    BootstrapDone,
    ExecuteOnNode,
    HeadAddressKnown,
    NodeInstance,
    Preempted,
    Provision,
    TaskResult,
)
from skyward.infra.ssh_actor import ConnectionFailed, ConnectionLost, ConnectionRestored, PortReForwarded

if TYPE_CHECKING:
    from casty import ClusterClient

    from skyward.api.plugin import AppLifecycle, ProcessLifecycle


@dataclass(frozen=True, slots=True)
class _PollResult:
    instance: Any | None = None
    cluster: Any | None = None


@dataclass(frozen=True, slots=True)
class _LocalInstallDone:
    instance: NodeInstance


@dataclass(frozen=True, slots=True)
class _PostBootstrapFailed:
    error: str


@dataclass(frozen=True, slots=True)
class _UserCodeSyncDone:
    instance: NodeInstance


@dataclass(frozen=True, slots=True)
class _Connected:
    transport_ref: ActorRef  # ActorRef[TransportMsg]
    local_port: int
    instance: NodeInstance | None = None


@dataclass(frozen=True, slots=True)
class _ConnectionFailed:
    error: str


@dataclass(frozen=True, slots=True)
class _WorkerStarted:
    local_port: int
    private_ip: str


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
class _SnapshotSaved:
    pass


@dataclass(frozen=True, slots=True)
class _SnapshotFailed:
    error: str


@dataclass(frozen=True, slots=True)
class _RemoteTaskDone:
    task_id: str
    value: Any
    node_id: int
    reply_to: ActorRef[Any]
    error: bool = False
    connection_error: bool = False


@dataclass(frozen=True, slots=True)
class JoinCluster:
    client: ClusterClient
    pool_info_json: str
    env_vars: dict[str, str]
    around_app_hooks: tuple[tuple[str, AppLifecycle], ...] = ()
    around_process_hooks: tuple[tuple[str, ProcessLifecycle], ...] = ()


@dataclass(frozen=True, slots=True)
class _WorkerDiscovered:
    worker_ref: ActorRef


@dataclass(frozen=True, slots=True)
class _WorkerDiscoveryFailed:
    error: str


@dataclass(frozen=True, slots=True)
class _EnvSetupDone:
    pass


@dataclass(frozen=True, slots=True)
class _EnvSetupFailed:
    error: str


type NodeMsg = (
    Provision
    | ExecuteOnNode
    | TaskResult
    | HeadAddressKnown
    | JoinCluster
    | Preempted
    | BootstrapDone
    | _PollResult
    | _LocalInstallDone
    | _PostBootstrapFailed
    | _UserCodeSyncDone
    | _Connected | _ConnectionFailed
    | _WorkerStarted | _WorkerFailed
    | _BootstrapUploaded | _BootstrapUploadFailed
    | _SnapshotSaved | _SnapshotFailed
    | _RemoteTaskDone
    | _WorkerDiscovered | _WorkerDiscoveryFailed
    | _EnvSetupDone | _EnvSetupFailed
    | ConnectionLost | ConnectionRestored | ConnectionFailed | PortReForwarded
    | Terminated
)
