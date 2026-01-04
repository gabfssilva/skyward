"""Instance allocation strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# =============================================================================
# Allocation Strategies
# =============================================================================


@dataclass(frozen=True, slots=True)
class _AllocationOnDemand:
    """Always use On-Demand instances."""

    def __repr__(self) -> str:
        return "Allocation.OnDemand"


class Allocation:
    """Instance allocation strategies.

    Usage:
        allocation='spot-if-available'    # Default: try Spot, fallback to On-Demand
        allocation='always-spot'          # Must be Spot, fail if unavailable
        allocation='on-demand'            # Always On-Demand
        allocation='cheapest'             # Compare all options, pick cheapest
        allocation=0.8                    # Shortcut for Allocation.Percent(spot=0.8)
        allocation=Allocation.Percent(spot=0.8)  # Minimum 80% Spot
    """

    @dataclass(frozen=True, slots=True)
    class AlwaysSpot:
        """Always use Spot instances. Fail if unavailable."""

        pass

    @dataclass(frozen=True, slots=True)
    class SpotIfAvailable:
        """Try Spot instances, fallback to On-Demand if unavailable."""

        pass

    @dataclass(frozen=True, slots=True)
    class Cheapest:
        """Choose cheapest option comparing spot and on-demand prices."""

        pass

    @dataclass(frozen=True, slots=True)
    class Percent:
        """Minimum percentage of Spot instances, rest On-Demand.

        Example: Allocation.Percent(spot=0.8) means at least 80% must be Spot.
        """

        spot: float  # 0.0 to 1.0

        def __post_init__(self) -> None:
            if not 0.0 <= self.spot <= 1.0:
                raise ValueError(f"spot must be between 0.0 and 1.0, got {self.spot}")

    # Singleton for On-Demand strategy
    OnDemand: _AllocationOnDemand = _AllocationOnDemand()


# String literals for convenience
type AllocationLiteral = Literal["always-spot", "spot-if-available", "on-demand", "cheapest"]

# Input type: what users can pass to allocation= parameter
type AllocationLike = (
    Allocation.AlwaysSpot
    | Allocation.SpotIfAvailable
    | Allocation.Cheapest
    | Allocation.Percent
    | _AllocationOnDemand
    | AllocationLiteral
    | float
)

# Normalized type (after normalize_allocation)
type NormalizedAllocation = (
    Allocation.AlwaysSpot
    | Allocation.SpotIfAvailable
    | Allocation.Cheapest
    | Allocation.Percent
    | _AllocationOnDemand
)


def normalize_allocation(allocation: AllocationLike) -> NormalizedAllocation:
    """Normalize allocation strategy from string/float to class instance.

    Args:
        allocation: Allocation strategy as string, float, or class.

    Returns:
        Normalized Allocation strategy instance.
    """
    match allocation:
        case float() as f:
            return Allocation.Percent(spot=f)
        case "always-spot":
            return Allocation.AlwaysSpot()
        case "spot-if-available":
            return Allocation.SpotIfAvailable()
        case "on-demand":
            return Allocation.OnDemand
        case "cheapest":
            return Allocation.Cheapest()
        case (
            Allocation.AlwaysSpot()
            | Allocation.SpotIfAvailable()
            | Allocation.Cheapest()
            | Allocation.Percent()
            | _AllocationOnDemand()
        ):
            return allocation
        case _:
            raise ValueError(
                f"Invalid allocation strategy: {allocation!r}. "
                "Use 'always-spot', 'spot-if-available', 'on-demand', 'cheapest', "
                "or a float (0.0-1.0)."
            )


# =============================================================================
# Fleet Allocation Strategies (EC2-specific)
# =============================================================================

AllocationStrategy = Literal[
    "price-capacity-optimized",  # Default: balance price and capacity
    "capacity-optimized",  # Prioritize pools with available capacity
    "lowest-price",  # Cheapest (more interruptions)
]
