"""Verda API response types.

TypedDicts for API responses - no conversion needed.
"""

from __future__ import annotations

import re
from typing import NotRequired, TypedDict

# =============================================================================
# Response Types
# =============================================================================


class CPUInfo(TypedDict):
    """CPU configuration."""

    number_of_cores: int


class MemoryInfo(TypedDict):
    """Memory configuration."""

    size_in_gigabytes: float


class GPUInfo(TypedDict):
    """GPU configuration."""

    description: str
    number_of_gpus: int


class GPUMemoryInfo(TypedDict):
    """GPU memory configuration."""

    size_in_gigabytes: float


class StorageInfo(TypedDict):
    """Storage configuration."""

    description: str


class InstanceTypeResponse(TypedDict):
    """Available instance type configuration."""

    instance_type: str
    cpu: CPUInfo
    memory: MemoryInfo
    gpu: NotRequired[GPUInfo]
    gpu_memory: NotRequired[GPUMemoryInfo]
    storage: NotRequired[StorageInfo]
    price_per_hour: str | None
    spot_price: str | None
    supported_os: list[str]


class InstanceResponse(TypedDict):
    """Instance status from API."""

    id: str
    hostname: str
    status: str
    ip: str
    is_spot: bool
    volume_ids: NotRequired[list[str]]
    os_volume_id: NotRequired[str]


class SSHKeyResponse(TypedDict):
    """SSH key from API."""

    id: str
    name: str
    fingerprint: str
    key: NotRequired[str]


class StartupScriptResponse(TypedDict):
    """Startup script from API."""

    id: str
    name: str
    script: str


class AvailabilityRegion(TypedDict):
    """Region availability info."""

    location_code: str
    availabilities: list[str]


# =============================================================================
# Helper functions for accessing typed data
# =============================================================================


GPU_MODEL_NORMALIZATIONS = {
    "Tesla V100": "V100",
    "RTX A6000": "A6000",
    "RTX 6000 Ada": "RTX6000Ada",
    "RTX PRO 6000": "RTXPRO6000",
}


def parse_gpu_model(description: str) -> str | None:
    """Parse GPU model from Verda description string.

    Args:
        description: GPU description (e.g., "8x H100 SXM5 80GB")

    Returns:
        Normalized GPU model (e.g., "H100") or None.
    """
    if not description:
        return None

    match = re.match(r"^\d+x\s+(.+?)\s+\d+GB$", description)
    if not match:
        return None

    model_raw = match.group(1)
    model_clean = re.sub(r"\s+SXM\d+", "", model_raw)

    if model_clean in GPU_MODEL_NORMALIZATIONS:
        return GPU_MODEL_NORMALIZATIONS[model_clean]

    return model_clean.replace(" ", "")


def get_vcpu(instance_type: InstanceTypeResponse) -> int:
    """Get vCPU count from instance type."""
    cpu = instance_type.get("cpu")
    return cpu.get("number_of_cores", 0) if cpu else 0


def get_memory_gb(instance_type: InstanceTypeResponse) -> float:
    """Get memory in GB from instance type."""
    mem = instance_type.get("memory")
    return mem.get("size_in_gigabytes", 0) if mem else 0


def get_accelerator(instance_type: InstanceTypeResponse) -> str | None:
    """Get accelerator name from instance type."""
    gpu = instance_type.get("gpu")
    if not gpu:
        return None
    return parse_gpu_model(gpu.get("description", ""))


def get_accelerator_count(instance_type: InstanceTypeResponse) -> int:
    """Get accelerator count from instance type."""
    gpu = instance_type.get("gpu")
    return gpu.get("number_of_gpus", 0) if gpu else 0


def get_accelerator_memory_gb(instance_type: InstanceTypeResponse) -> float:
    """Get accelerator memory in GB from instance type."""
    gpu_mem = instance_type.get("gpu_memory")
    return gpu_mem.get("size_in_gigabytes", 0) if gpu_mem else 0


def get_price_on_demand(instance_type: InstanceTypeResponse) -> float | None:
    """Get on-demand price from instance type."""
    price = instance_type.get("price_per_hour")
    if price is None:
        return None
    try:
        return float(price)
    except (ValueError, TypeError):
        return None


def get_price_spot(instance_type: InstanceTypeResponse) -> float | None:
    """Get spot price from instance type."""
    price = instance_type.get("spot_price")
    if price is None:
        return None
    try:
        return float(price)
    except (ValueError, TypeError):
        return None


__all__ = [
    # Response types
    "CPUInfo",
    "MemoryInfo",
    "GPUInfo",
    "GPUMemoryInfo",
    "StorageInfo",
    "InstanceTypeResponse",
    "InstanceResponse",
    "SSHKeyResponse",
    "StartupScriptResponse",
    "AvailabilityRegion",
    # Helpers
    "parse_gpu_model",
    "get_vcpu",
    "get_memory_gb",
    "get_accelerator",
    "get_accelerator_count",
    "get_accelerator_memory_gb",
    "get_price_on_demand",
    "get_price_spot",
]
