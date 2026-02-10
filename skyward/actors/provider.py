from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from skyward.messages import (
    BootstrapRequested,
    ClusterRequested,
    InstanceMetadata,
    InstanceRequested,
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
    """Internal: cluster provisioning background task completed."""

    state: Any


type ProviderMsg = (
    ClusterRequested
    | InstanceRequested
    | BootstrapRequested
    | ShutdownRequested
    | InstanceReady
    | BootstrapDone
    | _ProvisioningDone
)
