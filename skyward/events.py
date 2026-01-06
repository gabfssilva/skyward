"""Algebraic Data Type (ADT) for Skyward events.

This module defines strongly-typed events for all provider phases:
- Provision: ProvisioningStarted, NetworkReady, InstanceLaunching, InstanceProvisioned
- Setup: BootstrapStarting, BootstrapProgress, BootstrapCompleted, InstanceReady
- Pool: PoolStarted, PoolReady, PoolStopping
- Execute: Metrics, LogLine, FunctionCall
- Shutdown: InstanceStopping
- Errors: Error

All instance-related events use ProvisionedInstance for consistent data:

    match event:
        case BootstrapCompleted(instance=inst):
            print(f"Node {inst.node} ({inst.instance_id}) ready")
            if inst.spot:
                print("  (spot instance)")

        case ProvisioningCompleted(instances=insts, region=region):
            spot = sum(1 for i in insts if i.spot)
            print(f"Provisioned {len(insts)} instances ({spot} spot) in {region}")
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from skyward.callback import emit

if TYPE_CHECKING:
    from skyward.types import InstanceSpec


# =============================================================================
# Enums and Base Types
# =============================================================================


class ProviderName(Enum):
    AWS = "AWS"
    DigitalOcean = "Digital Ocean"
    Verda = "Verda"
    VastAI = "Vast.ai"


@dataclass(frozen=True, slots=True)
class ProvisionedInstance:
    """Information about a provisioned instance.

    Used by all instance-related events for consistent data access.
    """

    instance_id: str
    node: int
    provider: ProviderName
    spot: bool
    spec: InstanceSpec | None = None
    ip: str | None = None


# =============================================================================
# Provision Phase Events
# =============================================================================


@dataclass(frozen=True, slots=True)
class ProvisioningStarted:
    """Provisioning phase has begun.

    Emitted at the start of provision(), indicating the pool
    is beginning to acquire cloud resources.
    """

    pass


@dataclass(frozen=True, slots=True)
class NetworkReady:
    """Network infrastructure is ready.

    Emitted after VPC, subnet, and security groups have been
    validated or created. Instances can now be launched.
    """

    region: str


@dataclass(frozen=True, slots=True)
class InstanceLaunching:
    """Instances being launched.

    When multiple candidates are provided, the provider will choose
    one with available capacity based on allocation strategy.
    """

    count: int
    candidates: tuple[InstanceSpec, ...]
    provider: ProviderName


@dataclass(frozen=True, slots=True)
class InstanceProvisioned:
    """Instance provisioned and running."""

    instance: ProvisionedInstance


@dataclass(frozen=True, slots=True)
class ProvisioningCompleted:
    """All instances have been provisioned."""

    instances: tuple[ProvisionedInstance, ...]
    provider: ProviderName
    region: str


@dataclass(frozen=True, slots=True)
class RegionAutoSelected:
    """Region was auto-selected due to availability.

    Emitted when the requested instance type is not available in the
    configured region, and a different region was automatically selected.
    """

    requested_region: str
    selected_region: str
    spec: InstanceSpec
    provider: ProviderName


# =============================================================================
# Setup Phase Events
# =============================================================================


@dataclass(frozen=True, slots=True)
class BootstrapStarting:
    """Bootstrap starting on instance."""

    instance: ProvisionedInstance


@dataclass(frozen=True, slots=True)
class BootstrapProgress:
    """Bootstrap step completed."""

    instance: ProvisionedInstance
    step: str  # "uv", "apt", "deps", "skyward", "server", "volumes"


@dataclass(frozen=True, slots=True)
class BootstrapCompleted:
    """Bootstrap finished on instance."""

    instance: ProvisionedInstance


@dataclass(frozen=True, slots=True)
class InstanceReady:
    """Instance is fully ready to receive work.

    Emitted after bootstrap AND cluster setup is complete for this instance.
    The instance can now execute tasks.
    """

    instance: ProvisionedInstance


# =============================================================================
# Execute Phase Events
# =============================================================================


@dataclass(frozen=True, slots=True)
class Metrics:
    """Runtime metrics from instance."""

    instance: ProvisionedInstance
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

    instance: ProvisionedInstance
    line: str
    timestamp: float


@dataclass(frozen=True, slots=True)
class FunctionCall:
    """Remote function call started.

    Emitted when a @compute-decorated function is invoked on a remote instance.
    Used by panel callback to show recent function calls in the event tree.
    """

    function_name: str
    instance: ProvisionedInstance
    timestamp: float


@dataclass(frozen=True, slots=True)
class FunctionResult:
    """Remote function call completed.

    Emitted when a @compute-decorated function returns on a remote instance.
    Used by panel callback to track function completion.
    """

    function_name: str
    instance: ProvisionedInstance
    timestamp: float
    duration_seconds: float
    success: bool = True
    error: str | None = None


# =============================================================================
# Shutdown Phase Events
# =============================================================================


@dataclass(frozen=True, slots=True)
class InstanceStopping:
    """Instance being stopped/destroyed."""

    instance: ProvisionedInstance


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
# Pool Events
# =============================================================================


@dataclass(frozen=True, slots=True)
class PoolReady:
    """Pool is fully ready for execution.

    Emitted after all instances are bootstrapped, all tunnels and
    connections are established, and cluster environment is configured.
    """

    instances: tuple[ProvisionedInstance, ...]
    tunnels: int
    connections: int
    total_duration_seconds: float


# =============================================================================
# Lifecycle Events
# =============================================================================


@dataclass(frozen=True, slots=True)
class PoolStarted:
    """Pool context has started.

    Emitted when ComputePool.__enter__() completes consumer registration.
    Consumers should initialize resources (threads, displays) when receiving this.
    """

    nodes: int = 0


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
    instance: ProvisionedInstance | None = None


# =============================================================================
# Union Type (ADT)
# =============================================================================

type SkywardEvent = (
    ProvisioningStarted
    | NetworkReady
    | InstanceLaunching
    | InstanceProvisioned
    | ProvisioningCompleted
    | RegionAutoSelected
    | BootstrapStarting
    | BootstrapProgress
    | BootstrapCompleted
    | InstanceReady
    | Metrics
    | LogLine
    | FunctionCall
    | FunctionResult
    | InstanceStopping
    | CostUpdate
    | CostFinal
    | PoolReady
    | PoolStarted
    | PoolStopping
    | Error
)

__all__ = [
    # Base types
    "ProviderName",
    "ProvisionedInstance",
    # Provision
    "ProvisioningStarted",
    "NetworkReady",
    "InstanceLaunching",
    "InstanceProvisioned",
    "ProvisioningCompleted",
    "RegionAutoSelected",
    # Setup
    "BootstrapStarting",
    "BootstrapProgress",
    "BootstrapCompleted",
    "InstanceReady",
    # Execute
    "Metrics",
    "LogLine",
    "FunctionCall",
    "FunctionResult",
    # Shutdown
    "InstanceStopping",
    # Cost
    "CostUpdate",
    "CostFinal",
    # Pool
    "PoolReady",
    "PoolStarted",
    "PoolStopping",
    # Errors
    "Error",
    # Union type
    "SkywardEvent",
    # Emit function (re-exported from callback)
    "emit",
]
