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
    # NVIDIA A100 40GB
    InstanceSpec("1A100.40G.6V", 6, 60, "A100-40GB", 1, 40),
    InstanceSpec("2A100.40G.12V", 12, 120, "A100-40GB", 2, 40),
    InstanceSpec("4A100.40G.24V", 24, 240, "A100-40GB", 4, 40),
    InstanceSpec("8A100.40G.48V", 48, 480, "A100-40GB", 8, 40),
    # NVIDIA A100 80GB
    InstanceSpec("1A100.80G.10V", 10, 120, "A100-80GB", 1, 80),
    InstanceSpec("2A100.80G.20V", 20, 240, "A100-80GB", 2, 80),
    InstanceSpec("4A100.80G.40V", 40, 480, "A100-80GB", 4, 80),
    InstanceSpec("8A100.80G.80V", 80, 960, "A100-80GB", 8, 80),
    # NVIDIA H100 SXM 80GB (with InfiniBand)
    InstanceSpec("1H100.SXM.80G.IB.45V", 45, 240, "H100-80GB", 1, 80),
    InstanceSpec("2H100.SXM.80G.IB.90V", 90, 480, "H100-80GB", 2, 80),
    InstanceSpec("4H100.SXM.80G.IB.180V", 180, 960, "H100-80GB", 4, 80),
    InstanceSpec("8H100.SXM.80G.IB.350V", 350, 1920, "H100-80GB", 8, 80),
    # NVIDIA H200 (141GB HBM3e)
    InstanceSpec("1H200.141G.45V", 45, 240, "H200", 1, 141),
    InstanceSpec("8H200.141G.350V", 350, 1920, "H200", 8, 141),
    # NVIDIA L40S (48GB)
    InstanceSpec("1L40S.48G.12V", 12, 120, "L40S", 1, 48),
    InstanceSpec("2L40S.48G.24V", 24, 240, "L40S", 2, 48),
    InstanceSpec("4L40S.48G.48V", 48, 480, "L40S", 4, 48),
    InstanceSpec("8L40S.48G.96V", 96, 960, "L40S", 8, 48),
    # NVIDIA GB200 (latest Blackwell)
    InstanceSpec("1GB200.192G.72V", 72, 480, "GB200", 1, 192),
]

# Standard CPU instances (no GPU)
STANDARD_INSTANCES: list[InstanceSpec] = [
    InstanceSpec("cpu-4v-16gb", 4, 16),
    InstanceSpec("cpu-8v-32gb", 8, 32),
    InstanceSpec("cpu-16v-64gb", 16, 64),
    InstanceSpec("cpu-32v-128gb", 32, 128),
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
