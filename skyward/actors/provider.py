from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from skyward.messages import (
    BootstrapRequested,
    ClusterRequested,
    InstanceMetadata,
    InstanceRequested,
    InstanceRunning,
    ShutdownRequested,
)


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
    | InstanceReady
    | BootstrapDone
    | _ProvisioningDone
    | _InstanceNowRunning
    | _InstanceWaitFailed
    | _BootstrapScriptDone
    | _BootstrapScriptFailed
)
