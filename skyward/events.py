"""Algebraic Data Type (ADT) for Skyward events.

This module defines strongly-typed events for all provider phases:
- Provision: InfraCreating, InfraCreated, InstanceLaunching, InstanceProvisioned
- Setup: BootstrapStarting, BootstrapProgress, BootstrapCompleted
- Execute: Metrics
- Shutdown: InstanceStopping
- Errors: Error

Use pattern matching to handle events in consumers:

    match event:
        case Metrics(cpu_percent=cpu, gpu_utilization=gpu):
            print(f"CPU: {cpu}%, GPU: {gpu}%")
        case BootstrapCompleted(instance_id=id):
            print(f"Instance {id} ready")
        case Error(message=msg):
            raise RuntimeError(msg)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from skyward.callback import emit

# =============================================================================
# Provision Phase Events
# =============================================================================

class ProviderName(Enum):
    AWS = 'AWS'
    DigitalOcean = 'Digital Ocean'
    Verda = 'Verda'


@dataclass(frozen=True, slots=True)
class InfraCreating:
    """Infrastructure provisioning started."""

    pass


@dataclass(frozen=True, slots=True)
class InfraCreated:
    """Infrastructure ready."""

    region: str


@dataclass(frozen=True, slots=True)
class InstanceLaunching:
    """Instances being launched."""

    count: int
    instance_type: str
    provider: ProviderName


@dataclass(frozen=True, slots=True)
class InstanceProvisioned:
    """Instance provisioned and ready."""

    instance_id: str
    node: int
    spot: bool
    provider: ProviderName
    ip: str | None = None
    instance_type: str | None = None
    price_on_demand: float | None = None
    price_spot: float | None = None
    billing_increment_minutes: int | None = None  # None = per-second billing


@dataclass(frozen=True, slots=True)
class ProvisioningCompleted:
    """All instances have been provisioned."""

    spot: int
    on_demand: int
    provider: ProviderName
    region: str
    instances: list[str]


@dataclass(frozen=True, slots=True)
class RegionAutoSelected:
    """Region was auto-selected due to availability.

    Emitted when the requested instance type is not available in the
    configured region, and a different region was automatically selected.
    """

    requested_region: str
    selected_region: str
    instance_type: str
    provider: ProviderName


# =============================================================================
# Setup Phase Events
# =============================================================================


@dataclass(frozen=True, slots=True)
class BootstrapStarting:
    """Bootstrap starting on instance."""

    instance_id: str


@dataclass(frozen=True, slots=True)
class BootstrapProgress:
    """Bootstrap step completed."""

    instance_id: str
    step: str  # "uv", "apt", "deps", "skyward", "server", "volumes"


@dataclass(frozen=True, slots=True)
class BootstrapCompleted:
    """Bootstrap finished on instance."""

    instance_id: str


# =============================================================================
# Execute Phase Events
# =============================================================================


@dataclass(frozen=True, slots=True)
class Metrics:
    """Runtime metrics from instance."""

    instance_id: str
    node: int
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_total_mb: float
    gpu_utilization: float | None = None
    gpu_memory_used_mb: float | None = None
    gpu_memory_total_mb: float | None = None
    gpu_temperature: float | None = None


@dataclass(frozen=True, slots=True)
class LogLine:
    """Log line from remote execution stdout."""

    node: int
    instance_id: str
    line: str
    timestamp: float


# =============================================================================
# Shutdown Phase Events
# =============================================================================


@dataclass(frozen=True, slots=True)
class InstanceStopping:
    """Instance being stopped/destroyed."""

    instance_id: str


# =============================================================================
# Cost Events
# =============================================================================


@dataclass(frozen=True, slots=True)
class CostUpdate:
    """Real-time cost update from CostConsumer."""

    elapsed_seconds: float
    accumulated_cost: float
    hourly_rate: float
    spot_count: int
    ondemand_count: int


@dataclass(frozen=True, slots=True)
class CostFinal:
    """Final cost summary at shutdown."""

    total_cost: float
    total_seconds: float
    hourly_rate: float
    spot_count: int
    ondemand_count: int
    savings_vs_ondemand: float


# =============================================================================
# Lifecycle Events
# =============================================================================


@dataclass(frozen=True, slots=True)
class PoolStarted:
    """Pool context has started.

    Emitted when ComputePool.__enter__() completes consumer registration.
    Consumers should initialize resources (threads, displays) when receiving this.
    """

    pass


@dataclass(frozen=True, slots=True)
class PoolStopping:
    """Pool context is stopping.

    Emitted before ComputePool.__exit__() cleans up.
    Consumers should finalize resources (stop threads, print final output).
    """

    pass


# =============================================================================
# Error Events
# =============================================================================


@dataclass(frozen=True, slots=True)
class Error:
    """Error occurred."""

    message: str
    instance_id: str | None = None


# =============================================================================
# Union Type (ADT)
# =============================================================================

type SkywardEvent = (
    InfraCreating
    | InfraCreated
    | InstanceLaunching
    | InstanceProvisioned
    | ProvisioningCompleted
    | RegionAutoSelected
    | BootstrapStarting
    | BootstrapProgress
    | BootstrapCompleted
    | Metrics
    | LogLine
    | InstanceStopping
    | CostUpdate
    | CostFinal
    | PoolStarted
    | PoolStopping
    | Error
)

__all__ = [
    # Provision
    "InfraCreating",
    "InfraCreated",
    "InstanceLaunching",
    "InstanceProvisioned",
    "ProvisioningCompleted",
    "RegionAutoSelected",
    "ProviderName",
    # Setup
    "BootstrapStarting",
    "BootstrapProgress",
    "BootstrapCompleted",
    # Execute
    "Metrics",
    "LogLine",
    # Shutdown
    "InstanceStopping",
    # Cost
    "CostUpdate",
    "CostFinal",
    # Lifecycle
    "PoolStarted",
    "PoolStopping",
    # Errors
    "Error",
    # Union type
    "SkywardEvent",
    # Emit function (re-exported from callback)
    "emit",
]
