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

from collections.abc import Callable
from dataclasses import dataclass

# =============================================================================
# Provision Phase Events
# =============================================================================


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


@dataclass(frozen=True, slots=True)
class InstanceProvisioned:
    """Instance provisioned and ready."""

    instance_id: str
    node: int
    spot: bool
    ip: str | None = None
    instance_type: str | None = None


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

SkywardEvent = (
    InfraCreating
    | InfraCreated
    | InstanceLaunching
    | InstanceProvisioned
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

# Type alias for event callback
EventCallback = Callable[[SkywardEvent], None] | None


__all__ = [
    # Provision
    "InfraCreating",
    "InfraCreated",
    "InstanceLaunching",
    "InstanceProvisioned",
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
    # Callback type
    "EventCallback",
]
