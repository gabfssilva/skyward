"""Specification dataclasses for pool and image configuration.

These are the immutable configuration objects that define what
the user wants. Components use these specs to provision resources.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from .image import Image

if TYPE_CHECKING:
    from .accelerators import Accelerator
    from .messages import ProviderName


# =============================================================================
# Allocation Strategy
# =============================================================================

type AllocationStrategy = Literal[
    "spot",  # Always use spot instances
    "on-demand",  # Always use on-demand instances
    "spot-if-available",  # Try spot, fallback to on-demand (default)
    "cheapest",  # Compare prices and pick cheapest
]

type Architecture = Literal["x86_64", "arm64"]


# =============================================================================
# PoolSpec
# =============================================================================


@dataclass(frozen=True, slots=True)
class PoolSpec:
    """Pool specification - what the user wants.

    Defines the cluster configuration including number of nodes,
    hardware requirements, and instance allocation strategy.

    Args:
        nodes: Number of nodes in the cluster.
        accelerator: GPU/accelerator type - either a string (e.g., "A100", "H100")
            or an Accelerator instance from skyward.accelerators.
        region: Cloud region for instances.
        vcpus: Minimum vCPUs per node.
        memory_gb: Minimum memory in GB per node.
        architecture: CPU architecture ("x86_64" or "arm64"), or None for cheapest.
        allocation: Spot/on-demand strategy.
        image: Environment specification.
        ttl: Auto-shutdown timeout in seconds (0 = disabled).
        provider: Override provider (usually inferred from context).

    Example:
        >>> spec = PoolSpec(
        ...     nodes=4,
        ...     accelerator="H100",
        ...     region="us-east-1",
        ...     allocation="spot-if-available",
        ...     image=Image(pip=["torch"]),
        ... )

        >>> from skyward.accelerators import H100
        >>> spec = PoolSpec(
        ...     nodes=4,
        ...     accelerator=H100(count=8),
        ...     region="us-east-1",
        ... )
    """

    nodes: int
    accelerator: Accelerator | str | None
    region: str
    vcpus: int | None = None
    memory_gb: int | None = None
    architecture: Architecture | None = None
    allocation: AllocationStrategy = "spot-if-available"
    image: Image = field(default_factory=Image)
    ttl: int = 600
    concurrency: int = 1
    provider: ProviderName | None = None
    max_hourly_cost: float | None = None

    def __post_init__(self) -> None:
        if self.nodes < 1:
            raise ValueError(f"nodes must be >= 1, got {self.nodes}")

    @property
    def accelerator_name(self) -> str | None:
        """Get the canonical accelerator name for provider matching.

        Returns:
            Accelerator name string for use in provider instance selection,
            or None if no accelerator specified.
        """
        match self.accelerator:
            case None:
                return None
            case str(name):
                return name
            case accel:
                return accel.name

    @property
    def accelerator_count(self) -> int:
        """Get the number of accelerators per node.

        Returns:
            Number of accelerators (1 if using string or not specified, 0 if None).
        """
        match self.accelerator:
            case None:
                return 0
            case str():
                return 1
            case accel:
                return accel.count


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "AllocationStrategy",
    "Architecture",
    "PoolSpec",
]
