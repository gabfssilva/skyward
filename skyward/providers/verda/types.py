"""Verda API response types.

TypedDicts for API responses - no conversion needed.
"""

from __future__ import annotations

import re
from typing import NotRequired, TypedDict


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


def select_os_image(supported_os: list[str], cuda_max: str = "") -> str:
    """Select the best CUDA OS image for a GPU.

    Picks the highest CUDA version within the GPU's supported range,
    on the newest Ubuntu, preferring images without docker/cluster suffixes.

    Parameters
    ----------
    supported_os
        Image names from the Verda instance-types API.
    cuda_max
        Maximum CUDA version the GPU supports (e.g. ``"13.1"``).
        Empty string means no upper bound.
    """
    if not supported_os:
        return "ubuntu-22.04"

    max_ver = _parse_ver(cuda_max) if cuda_max else (99, 99)

    def _is_candidate(img: str) -> bool:
        lower = img.lower()
        if not lower.startswith("ubuntu-"):
            return False
        if any(x in lower for x in ("kubernetes", "jupyter", "cluster")):
            return False
        ver = _parse_cuda_ver(img)
        return ver is not None and ver <= max_ver

    candidates = sorted(filter(_is_candidate, supported_os), key=_image_rank, reverse=True)
    if candidates:
        return candidates[0]

    cuda_any = sorted(
        (img for img in supported_os if "cuda" in img.lower()),
        key=_image_rank, reverse=True,
    )
    if cuda_any:
        return cuda_any[0]

    return supported_os[0]


def _parse_ver(s: str) -> tuple[int, int]:
    parts = s.split(".")
    return (int(parts[0]), int(parts[1])) if len(parts) >= 2 else (int(parts[0]), 0)


def _parse_cuda_ver(img: str) -> tuple[int, int] | None:
    m = re.search(r"cuda-?(\d+)\.(\d+)", img.lower())
    return (int(m.group(1)), int(m.group(2))) if m else None


def _image_rank(img: str) -> tuple[int, ...]:
    cuda = _parse_cuda_ver(img) or (0, 0)
    m = re.search(r"ubuntu-(\d+)\.(\d+)", img.lower())
    ub = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    no_docker = 0 if "docker" in img.lower() else 1
    return (*cuda, *ub, no_docker)
