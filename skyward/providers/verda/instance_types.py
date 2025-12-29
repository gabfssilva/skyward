"""Verda instance type selection based on resource requirements.

Instance types from Verda (formerly DataCrunch):
https://verda.com/
https://api.datacrunch.io/v1/docs
"""

from __future__ import annotations

from dataclasses import dataclass

from skyward.accelerator import Accelerator


@dataclass(frozen=True)
class InstanceSpec:
    """Specification for a Verda instance type."""

    slug: str
    vcpu: int
    memory_gb: float
    accelerator: Accelerator = None
    accelerator_count: int = 0
    accelerator_memory_gb: float = 0


# GPU Instance Types (sorted by GPU tier)
# Format: {count}{GPU}.{vCPU}V or similar patterns
# See: https://api.datacrunch.io/v1/docs -> GET /instance-types
ACCELERATOR_INSTANCES: list[InstanceSpec] = [
    # NVIDIA V100 (16GB)
    InstanceSpec("1V100.6V", 6, 22, "V100", 1, 16),
    InstanceSpec("2V100.10V", 10, 44, "V100", 2, 16),
    InstanceSpec("4V100.20V", 20, 88, "V100", 4, 16),
    InstanceSpec("8V100.48V", 48, 176, "V100", 8, 16),
    # NVIDIA A100 40GB (SXM variant)
    InstanceSpec("1A100.40S.22V", 22, 60, "A100-40GB", 1, 40),
    InstanceSpec("8A100.40S.176V", 176, 480, "A100-40GB", 8, 40),
    # NVIDIA A100 80GB
    InstanceSpec("1A100.22V", 22, 120, "A100-80GB", 1, 80),
    InstanceSpec("2A100.44V", 44, 240, "A100-80GB", 2, 80),
    InstanceSpec("4A100.88V", 88, 480, "A100-80GB", 4, 80),
    InstanceSpec("8A100.176V", 176, 960, "A100-80GB", 8, 80),
    # NVIDIA H100 SXM 80GB
    InstanceSpec("1H100.80S.30V", 30, 240, "H100-80GB", 1, 80),
    InstanceSpec("1H100.80S.32V", 32, 240, "H100-80GB", 1, 80),
    InstanceSpec("2H100.80S.80V", 80, 480, "H100-80GB", 2, 80),
    InstanceSpec("4H100.80S.176V", 176, 960, "H100-80GB", 4, 80),
    InstanceSpec("8H100.80S.176V", 176, 1920, "H100-80GB", 8, 80),
    # NVIDIA H200 (141GB HBM3e, SXM variant)
    InstanceSpec("1H200.141S.44V", 44, 240, "H200", 1, 141),
    InstanceSpec("2H200.141S.88V", 88, 480, "H200", 2, 141),
    InstanceSpec("4H200.141S.176V", 176, 960, "H200", 4, 141),
    InstanceSpec("8H200.141S.176V", 176, 1920, "H200", 8, 141),
    # NVIDIA L40S (48GB)
    InstanceSpec("1L40S.20V", 20, 120, "L40S", 1, 48),
    InstanceSpec("2L40S.40V", 40, 240, "L40S", 2, 48),
    InstanceSpec("4L40S.80V", 80, 480, "L40S", 4, 48),
    InstanceSpec("8L40S.160V", 160, 960, "L40S", 8, 48),
    # NVIDIA RTX 6000 Ada (48GB)
    InstanceSpec("1RTX6000ADA.10V", 10, 60, "RTX6000-ADA", 1, 48),
    InstanceSpec("2RTX6000ADA.20V", 20, 120, "RTX6000-ADA", 2, 48),
    InstanceSpec("4RTX6000ADA.40V", 40, 240, "RTX6000-ADA", 4, 48),
    InstanceSpec("8RTX6000ADA.80V", 80, 480, "RTX6000-ADA", 8, 48),
    # NVIDIA RTX PRO 6000 (48GB)
    InstanceSpec("1RTXPRO6000.30V", 30, 120, "RTXPRO6000", 1, 48),
    InstanceSpec("2RTXPRO6000.60V", 60, 240, "RTXPRO6000", 2, 48),
    InstanceSpec("4RTXPRO6000.120V", 120, 480, "RTXPRO6000", 4, 48),
    InstanceSpec("8RTXPRO6000.240V", 240, 960, "RTXPRO6000", 8, 48),
    # NVIDIA A6000 (48GB)
    InstanceSpec("1A6000.10V", 10, 60, "A6000", 1, 48),
    InstanceSpec("2A6000.20V", 20, 120, "A6000", 2, 48),
    InstanceSpec("4A6000.40V", 40, 240, "A6000", 4, 48),
    InstanceSpec("8A6000.80V", 80, 480, "A6000", 8, 48),
    # NVIDIA B200 (Blackwell)
    InstanceSpec("1B200.30V", 30, 240, "B200", 1, 192),
    InstanceSpec("2B200.60V", 60, 480, "B200", 2, 192),
    InstanceSpec("4B200.120V", 120, 960, "B200", 4, 192),
    InstanceSpec("8B200.240V", 240, 1920, "B200", 8, 192),
    # NVIDIA B300 (Blackwell)
    InstanceSpec("1B300.30V", 30, 240, "B300", 1, 288),
    InstanceSpec("2B300.60V", 60, 480, "B300", 2, 288),
    InstanceSpec("4B300.120V", 120, 960, "B300", 4, 288),
    InstanceSpec("8B300.240V", 240, 1920, "B300", 8, 288),
    # NVIDIA GB300 (Grace Blackwell)
    InstanceSpec("1GB300.36V", 36, 240, "GB300", 1, 288),
    InstanceSpec("2GB300.72V", 72, 480, "GB300", 2, 288),
    InstanceSpec("4GB300.144V", 144, 960, "GB300", 4, 288),
]

# Standard CPU instances (no GPU)
STANDARD_INSTANCES: list[InstanceSpec] = [
    InstanceSpec("CPU.4V.16G", 4, 16),
    InstanceSpec("CPU.8V.32G", 8, 32),
    InstanceSpec("CPU.16V.64G", 16, 64),
    InstanceSpec("CPU.32V.128G", 32, 128),
    InstanceSpec("CPU.64V.256G", 64, 256),
    InstanceSpec("CPU.96V.384G", 96, 384),
    InstanceSpec("CPU.120V.480G", 120, 480),
    InstanceSpec("CPU.180V.720G", 180, 720),
    InstanceSpec("CPU.360V.1440G", 360, 1440),
]


def select_instance_type(
    cpu: int,
    memory_mb: int,
    accelerator: Accelerator = None,
) -> InstanceSpec:
    """Select the smallest instance type that meets requirements.

    Args:
        cpu: Required number of vCPUs.
        memory_mb: Required memory in MB.
        accelerator: Required accelerator type (e.g., "H100-80GB", "A100-80GB").

    Returns:
        InstanceSpec with the selected instance type.

    Raises:
        ValueError: If no instance type meets the requirements.
    """
    memory_gb = memory_mb / 1024

    if accelerator:
        accel_normalized = _normalize_accelerator(accelerator)

        candidates = [
            spec
            for spec in ACCELERATOR_INSTANCES
            if spec.accelerator
            and _normalize_accelerator(spec.accelerator) == accel_normalized
            and spec.vcpu >= cpu
            and spec.memory_gb >= memory_gb
        ]
        if not candidates:
            available = sorted(
                {spec.accelerator for spec in ACCELERATOR_INSTANCES if spec.accelerator}
            )
            raise ValueError(
                f"No Verda instance type found for {accelerator} with {cpu} vCPU, {memory_gb}GB RAM. "
                f"Available accelerator types: {available}"
            )
        return candidates[0]

    # For standard instances, find the smallest that meets requirements
    candidates = [
        spec
        for spec in STANDARD_INSTANCES
        if spec.vcpu >= cpu and spec.memory_gb >= memory_gb
    ]
    if not candidates:
        max_vcpu = max(spec.vcpu for spec in STANDARD_INSTANCES)
        max_mem = max(spec.memory_gb for spec in STANDARD_INSTANCES)
        raise ValueError(
            f"No Verda instance type found for {cpu} vCPU, {memory_gb}GB RAM. "
            f"Maximum available: {max_vcpu} vCPU, {max_mem}GB RAM"
        )

    return candidates[0]


def _normalize_accelerator(accelerator: str) -> str:
    """Normalize accelerator name for comparison."""
    return accelerator.upper().replace("-", "").replace("_", "")


def get_gpu_image(accelerator: Accelerator, accelerator_count: int = 1) -> str:
    """Get the appropriate GPU image for the accelerator type.

    Verda provides CUDA-enabled Ubuntu images.

    Args:
        accelerator: The accelerator type.
        accelerator_count: Number of GPUs.

    Returns:
        Image slug for GPU instance.
    """
    # Verda uses CUDA-enabled Ubuntu images
    # Latest available: ubuntu-24.04-cuda-12.8-open-docker
    return "ubuntu-24.04-cuda-12.8-open-docker"


def get_standard_image() -> str:
    """Get the standard Ubuntu image for CPU instances."""
    return "ubuntu-24.04"
