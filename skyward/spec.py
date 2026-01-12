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
    from .events import ProviderName


# =============================================================================
# Allocation Strategy
# =============================================================================

type AllocationStrategy = Literal[
    "spot",  # Always use spot instances
    "on-demand",  # Always use on-demand instances
    "spot-if-available",  # Try spot, fallback to on-demand (default)
    "cheapest",  # Compare prices and pick cheapest
]


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
    accelerator: Accelerator | str
    region: str
    allocation: AllocationStrategy = "spot-if-available"
    image: Image = field(default_factory=Image)
    ttl: int = 0
    provider: ProviderName | None = None
    max_hourly_cost: float | None = None  # USD/hr for entire cluster

    def __post_init__(self) -> None:
        if self.nodes < 1:
            raise ValueError(f"nodes must be >= 1, got {self.nodes}")

    @property
    def accelerator_name(self) -> str:
        """Get the canonical accelerator name for provider matching.

        Returns:
            Accelerator name string for use in provider instance selection.
        """
        if isinstance(self.accelerator, str):
            return self.accelerator
        return self.accelerator.name

    @property
    def accelerator_count(self) -> int:
        """Get the number of accelerators per node.

        Returns:
            Number of accelerators (1 if using string or not specified).
        """
        if isinstance(self.accelerator, str):
            return 1
        return self.accelerator.count


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "AllocationStrategy",
    "PoolSpec",
]
