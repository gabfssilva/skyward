"""AWS provider types and constants.

Dataclasses, type definitions, and GPU model mappings used across the
AWS provider modules.
"""

from __future__ import annotations

from dataclasses import dataclass

_GPU_MODEL_BY_MANUFACTURER: dict[str, dict[str, str]] = {
    "nvidia": {
        "t4": "T4",
        "t4g": "T4G",
        "a10g": "A10G",
        "a100": "A100",
        "h100": "H100",
        "h200": "H200",
        "l4": "L4",
        "l40s": "L40S",
        "v100": "V100",
        "k80": "K80",
        "m60": "M60",
    },
    "amd": {
        "radeon-pro-v520": "Radeon Pro V520",
    },
    "aws": {
        "trainium": "Trainium1",
        "trainium2": "Trainium2",
        "inferentia": "Inferentia1",
        "inferentia2": "Inferentia2",
    },
}


@dataclass(frozen=True, slots=True)
class InstanceResources:
    instance_type: str
    vcpus: int
    memory_mb: int
    architecture: str
    gpu_count: int = 0
    gpu_model: str = ""
    gpu_vram_mb: int = 0
    network_bandwidth_gbps: float = 0.0
    instance_storage_gb: int = 0
    instance_storage_type: str = ""

    @property
    def memory_gb(self) -> float:
        return self.memory_mb / 1024

    @property
    def gpu_vram_gb(self) -> float:
        return self.gpu_vram_mb / 1024


@dataclass(frozen=True, slots=True)
class InstanceSpec:
    instance_type: str
    region: str
    vcpus: int
    memory_gb: float
    architecture: str
    gpu_count: int = 0
    gpu_model: str = ""
    gpu_vram_gb: float = 0.0
    network_bandwidth_gbps: float = 0.0
    ondemand_price: float | None = None
    spot_price: float | None = None


@dataclass(frozen=True, slots=True)
class AWSOfferSpecific:
    ami: str


@dataclass(frozen=True, slots=True)
class AWSResources:
    instance_profile_arn: str
    security_group_id: str
    subnet_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AWSSpecific:
    resources: AWSResources
    ssh_key_name: str
    pinned_az: str | None = None


@dataclass(frozen=True, slots=True)
class IAMStatement:
    """Single IAM policy statement."""

    actions: tuple[str, ...]
    resources: tuple[str, ...]
