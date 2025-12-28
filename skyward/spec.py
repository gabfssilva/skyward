"""Spot instance strategies and allocation configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# --- Spot Strategies ---


@dataclass(frozen=True, slots=True)
class _SpotNever:
    """Never use Spot instances. Always On-Demand."""

    def __repr__(self) -> str:
        return "Spot.Never"


class Spot:
    """Spot instance strategies.

    Usage:
        spot=Spot.IfAvailable()  # Default: try Spot, fallback to On-Demand
        spot=Spot.Always(retries=5, interval=2.0)  # Must be Spot, error if unavailable
        spot=Spot.Never  # Always On-Demand
        spot=Spot.Percent(0.8)  # Minimum 80% Spot, rest On-Demand
        spot=0.8  # Shortcut for Spot.Percent(0.8)

        # String shortcuts (use default retries/interval):
        spot="always"
        spot="if-available"
        spot="never"
    """

    @dataclass(frozen=True, slots=True)
    class Always:
        """Always use Spot instances. Raises SpotCapacityError after retries."""

        retries: int = 10
        interval: float = 1.0  # seconds between retries

    @dataclass(frozen=True, slots=True)
    class IfAvailable:
        """Try Spot instances, fallback to On-Demand if unavailable."""

        retries: int = 10
        interval: float = 1.0  # seconds between retries

    @dataclass(frozen=True, slots=True)
    class Percent:
        """Minimum percentage of Spot instances, rest On-Demand.

        Example: Spot.Percent(0.8) means at least 80% must be Spot.
        If minimum cannot be met, raises SpotMinimumNotMetError.
        """

        percentage: float  # 0.0 to 1.0

        def __post_init__(self) -> None:
            if not 0.0 <= self.percentage <= 1.0:
                raise ValueError(f"percentage must be between 0.0 and 1.0, got {self.percentage}")

    # Singleton instance for Never strategy
    Never: _SpotNever = _SpotNever()


# String literals for convenience
SpotLiteral = Literal["always", "if-available", "never"]

# Input type: what users can pass to spot= parameter
# Accepts: "always", "if-available", "never", 0.8, Spot.Always(), Spot.IfAvailable(), Spot.Percent(0.8), Spot.Never
SpotLike = Spot.Always | Spot.IfAvailable | Spot.Percent | _SpotNever | SpotLiteral | float

# Alias for backwards compatibility
SpotStrategy = SpotLike

# Normalized type (after normalize_spot)
NormalizedSpot = Spot.Always | Spot.IfAvailable | Spot.Percent | _SpotNever


def normalize_spot(spot: SpotLike) -> NormalizedSpot:
    """Normalize spot strategy from string/float to class instance.

    Args:
        spot: Spot strategy as string, float, or class.

    Returns:
        Normalized Spot strategy instance.
    """
    if isinstance(spot, float):
        return Spot.Percent(spot)
    if isinstance(spot, str):
        match spot:
            case "always":
                return Spot.Always()
            case "if-available":
                return Spot.IfAvailable()
            case "never":
                return Spot.Never
            case _:
                raise ValueError(f"Invalid spot strategy: {spot!r}. Use 'always', 'if-available', 'never', or a float (0.0-1.0).")
    return spot


class SpotCapacityError(Exception):
    """Raised when Spot capacity is unavailable and strategy is Spot.Always."""

    def __init__(self, retries: int, instance_type: str) -> None:
        super().__init__(
            f"Spot capacity unavailable for {instance_type} after {retries} retries"
        )
        self.retries = retries
        self.instance_type = instance_type


class SpotMinimumNotMetError(Exception):
    """Raised when minimum Spot percentage cannot be met."""

    def __init__(self, required: int, got: int, total: int) -> None:
        percentage = (got / total * 100) if total > 0 else 0
        super().__init__(
            f"Spot minimum not met: required {required}/{total}, got {got} ({percentage:.0f}%)"
        )
        self.required = required
        self.got = got
        self.total = total


# --- Allocation Strategies (EC2 Fleet) ---

AllocationStrategy = Literal[
    "price-capacity-optimized",  # Default: balance price and capacity
    "capacity-optimized",  # Prioritize pools with available capacity
    "lowest-price",  # Cheapest (more interruptions)
]
