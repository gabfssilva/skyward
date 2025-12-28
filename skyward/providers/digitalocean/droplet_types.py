"""DigitalOcean droplet type selection based on resource requirements."""

from __future__ import annotations

from dataclasses import dataclass

from skyward.accelerator import Accelerator


@dataclass(frozen=True)
class DropletSpec:
    """Specification for a DigitalOcean droplet type."""

    slug: str
    vcpu: int
    memory_gb: float
    accelerator: Accelerator = None
    accelerator_count: int = 0
    accelerator_memory_gb: float = 0


# Standard compute droplets (sorted by resources ascending)
# https://docs.digitalocean.com/products/droplets/concepts/choosing-a-plan/
STANDARD_DROPLETS: list[DropletSpec] = [
    # Basic Droplets
    DropletSpec("s-1vcpu-512mb-10gb", 1, 0.5),
    DropletSpec("s-1vcpu-1gb", 1, 1),
    DropletSpec("s-1vcpu-2gb", 1, 2),
    DropletSpec("s-2vcpu-2gb", 2, 2),
    DropletSpec("s-2vcpu-4gb", 2, 4),
    DropletSpec("s-4vcpu-8gb", 4, 8),
    DropletSpec("s-8vcpu-16gb", 8, 16),
    # General Purpose Droplets
    DropletSpec("g-2vcpu-8gb", 2, 8),
    DropletSpec("g-4vcpu-16gb", 4, 16),
    DropletSpec("g-8vcpu-32gb", 8, 32),
    DropletSpec("g-16vcpu-64gb", 16, 64),
    DropletSpec("g-32vcpu-128gb", 32, 128),
    DropletSpec("g-40vcpu-160gb", 40, 160),
    # CPU-Optimized Droplets
    DropletSpec("c-2", 2, 4),
    DropletSpec("c-4", 4, 8),
    DropletSpec("c-8", 8, 16),
    DropletSpec("c-16", 16, 32),
    DropletSpec("c-32", 32, 64),
    # Memory-Optimized Droplets
    DropletSpec("m-2vcpu-16gb", 2, 16),
    DropletSpec("m-4vcpu-32gb", 4, 32),
    DropletSpec("m-8vcpu-64gb", 8, 64),
    DropletSpec("m-16vcpu-128gb", 16, 128),
    DropletSpec("m-24vcpu-192gb", 24, 192),
    DropletSpec("m-32vcpu-256gb", 32, 256),
]

# GPU Droplets
# https://docs.digitalocean.com/products/droplets/details/gpu/
ACCELERATOR_DROPLETS: list[DropletSpec] = [
    # NVIDIA H100 (80GB HBM3) - Single and Multi-GPU
    DropletSpec("gpu-h100x1-80gb", 20, 240, "H100-80GB", 1, 80),
    DropletSpec("gpu-h100x8-640gb", 160, 1920, "H100-80GB", 8, 80),
    # NVIDIA H200 (141GB HBM3e) - Not yet available in all regions
    DropletSpec("gpu-h200x1-141gb", 20, 240, "H200", 1, 141),
    DropletSpec("gpu-h200x8-1128gb", 160, 1920, "H200", 8, 141),
    # NVIDIA L40S (48GB GDDR6)
    DropletSpec("gpu-l40sx1-48gb", 14, 120, "L40S", 1, 48),
    DropletSpec("gpu-l40sx4-192gb", 56, 480, "L40S", 4, 48),
    DropletSpec("gpu-l40sx8-384gb", 112, 960, "L40S", 8, 48),
    # NVIDIA RTX 4090 (24GB GDDR6X) - Consumer-grade but available
    DropletSpec("gpu-rtx4090x1-24gb", 8, 30, "RTX4090", 1, 24),
    # AMD MI300X (192GB HBM3) - Limited availability
    DropletSpec("gpu-mi300x1-192gb", 24, 384, "MI300X", 1, 192),
]


def select_droplet_type(
    cpu: int,
    memory_mb: int,
    accelerator: Accelerator = None,
) -> DropletSpec:
    """Select the smallest droplet type that meets requirements.

    Args:
        cpu: Required number of vCPUs.
        memory_mb: Required memory in MB.
        accelerator: Required accelerator type (e.g., "H100", "L40S").

    Returns:
        DropletSpec with the selected droplet type.

    Raises:
        ValueError: If no droplet type meets the requirements.
    """
    memory_gb = memory_mb / 1024

    if accelerator:
        # Normalize accelerator name for matching
        accel_normalized = _normalize_accelerator(accelerator)

        candidates = [
            spec
            for spec in ACCELERATOR_DROPLETS
            if spec.accelerator
            and _normalize_accelerator(spec.accelerator) == accel_normalized
            and spec.vcpu >= cpu
            and spec.memory_gb >= memory_gb
        ]
        if not candidates:
            available = sorted(
                {spec.accelerator for spec in ACCELERATOR_DROPLETS if spec.accelerator}
            )
            raise ValueError(
                f"No droplet type found for {accelerator} with {cpu} vCPU, {memory_gb}GB RAM. "
                f"Available accelerator types: {available}"
            )
        return candidates[0]

    # For standard droplets, find the smallest that meets requirements
    candidates = [
        spec
        for spec in STANDARD_DROPLETS
        if spec.vcpu >= cpu and spec.memory_gb >= memory_gb
    ]
    if not candidates:
        max_vcpu = max(spec.vcpu for spec in STANDARD_DROPLETS)
        max_mem = max(spec.memory_gb for spec in STANDARD_DROPLETS)
        raise ValueError(
            f"No droplet type found for {cpu} vCPU, {memory_gb}GB RAM. "
            f"Maximum available: {max_vcpu} vCPU, {max_mem}GB RAM"
        )

    # Return the first matching spec (list is sorted by size)
    return candidates[0]


def _normalize_accelerator(accelerator: str) -> str:
    """Normalize accelerator name for comparison."""
    return accelerator.upper().replace("-", "").replace("_", "")


def get_gpu_image(accelerator: Accelerator, accelerator_count: int = 1) -> str:
    """Get the appropriate GPU image for the accelerator type.

    DigitalOcean GPU Droplets require specific GPU-enabled images.
    See: https://docs.digitalocean.com/products/droplets/getting-started/recommended-gpu-setup/

    Args:
        accelerator: The accelerator type (e.g., "L40S", "H100-80GB", "MI300X").
        accelerator_count: Number of GPUs (1 or 8).

    Returns:
        Image slug for GPU droplet.
    """
    # AMD GPUs use a different base image
    if accelerator and "MI3" in accelerator.upper():
        return "gpu-amd-base"

    # NVIDIA GPUs - 8x configurations
    if accelerator_count == 8:
        return "gpu-h100x8-base"

    # NVIDIA GPUs - single GPU (all types use same base image)
    return "gpu-h100x1-base"


def get_standard_image() -> str:
    """Get the standard Ubuntu image for CPU droplets."""
    return "ubuntu-24-04-x64"
