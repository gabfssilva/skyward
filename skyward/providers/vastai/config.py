"""Vast.ai provider configuration.

Immutable configuration dataclass for Vast.ai provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class VastAI:
    """Vast.ai provider configuration.

    Vast.ai is a GPU marketplace with dynamic offers from various hosts.
    Unlike traditional cloud providers, instances are Docker containers
    running on marketplace hosts with varying reliability.

    SSH keys are automatically detected from ~/.ssh/id_ed25519.pub or
    ~/.ssh/id_rsa.pub and registered on Vast.ai if needed.

    Example:
        >>> from skyward.providers.vastai import VastAI
        >>> config = VastAI(min_reliability=0.95, geolocation="US")

    Args:
        api_key: Vast.ai API key. Falls back to VAST_API_KEY env var.
        min_reliability: Minimum host reliability score (0.0-1.0).
        min_cuda: Minimum CUDA version (e.g., 12.0). Filters out old/broken hosts.
        geolocation: Preferred region/country code (e.g., "US", "EU").
        bid_multiplier: For spot pricing, multiply min bid by this.
        instance_timeout: Auto-shutdown in seconds (safety timeout).
        docker_image: Base Docker image for containers.
        disk_gb: Disk space in GB.
        use_overlay: Enable overlay networking for multi-node clusters.
        overlay_timeout: Timeout for overlay operations in seconds.
    """

    api_key: str | None = None
    min_reliability: float = 0.95
    min_cuda: float = 12.0
    geolocation: str | None = None
    bid_multiplier: float = 1.2
    instance_timeout: int = 300
    docker_image: str | None = None
    disk_gb: int = 100
    use_overlay: bool = True
    overlay_timeout: int = 120

    @classmethod
    def ubuntu(
        cls,
        version: Literal["22.04", "24.04", "26.04"] | str = "24.04",
        cuda: Literal["12.9.1", "13.1.0", "13.0.1"] | str = "12.9.1",
        cuda_dist: Literal["devel", "runtime"] = "runtime",
    ) -> str:
        """Generate NVIDIA CUDA Docker image name."""
        return f"nvcr.io/nvidia/cuda:{cuda}-{cuda_dist}-ubuntu{version}"


# =============================================================================
# Exports
# =============================================================================

__all__ = ["VastAI"]
