from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from skyward.api.spec import Architecture, PoolSpec

if TYPE_CHECKING:
    from skyward.accelerators import Accelerator

type InstanceStatus = Literal[
    "provisioning",
    "provisioned",
    "boostrapping",
    "boostrapped",
    "ready"
]


@dataclass(frozen=True, slots=True)
class InstanceType[S]:
    """Normalized machine type — cacheable, provider-agnostic hardware description."""
    name: str
    accelerator: Accelerator | None
    vcpus: float
    memory_gb: float
    architecture: Architecture
    specific: S


@dataclass(frozen=True, slots=True)
class Offer[S]:
    """Ephemeral availability + pricing for a specific instance type."""
    id: str
    instance_type: InstanceType
    spot_price: float | None
    on_demand_price: float | None
    billing_unit: Literal["second", "minute", "hour"]
    specific: S


@dataclass(frozen=True, slots=True)
class Instance:
    id: str
    status: InstanceStatus
    offer: Offer
    ip: str | None = None
    private_ip: str | None = None
    ssh_port: int = 22
    spot: bool = False
    region: str = ""


type ClusterStatus = Literal[
    "setting_up",
    "provisioning",
    "bootstrapping",
    "ready",
    "shutting_down",
    "destroyed",
]


@dataclass(frozen=True, slots=True)
class Cluster[S]:
    id: str
    status: ClusterStatus
    spec: PoolSpec
    offer: Offer
    ssh_key_path: str
    ssh_user: str
    use_sudo: bool
    shutdown_command: str
    specific: S
    instances: tuple[Instance, ...] = ()
    prebaked: bool = False
