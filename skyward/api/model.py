from dataclasses import dataclass
from typing import Literal

from skyward.api.spec import PoolSpec

type InstanceStatus = Literal[
    "provisioning",
    "provisioned",
    "boostrapping",
    "boostrapped",
    "ready"
]


@dataclass(frozen=True, slots=True)
class Instance:
    id: str
    status: InstanceStatus
    ip: str | None = None
    private_ip: str | None = None
    ssh_port: int = 22
    spot: bool = False
    instance_type: str = ""
    accelerator_count: int = 0
    accelerator_model: str = ""
    vcpus: float = 0
    memory_gb: float = 0.0
    accelerator_vram_gb: int = 0
    region: str = ""
    hourly_rate: float = 0.0
    on_demand_rate: float = 0.0
    billing_increment: int = 1


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
    ssh_key_path: str
    ssh_user: str
    use_sudo: bool
    shutdown_command: str
    specific: S
    instances: tuple[Instance, ...] = ()
    prebaked: bool = False
