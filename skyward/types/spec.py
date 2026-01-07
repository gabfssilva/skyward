"""Instance specifications and selection functions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from skyward.accelerators import AcceleratorCount, AcceleratorSpec
from skyward.core.exceptions import NoMatchingInstanceError
from skyward.types.core import Auto, Memory

__all__ = [
    "InstanceSpec",
    "parse_memory_mb",
    "select_instance",
    "select_instances",
]


@dataclass(frozen=True, slots=True)
class InstanceSpec:
    """Unified instance specification across all providers.

    Represents an available instance/droplet type with its specs.
    Provider-specific data (e.g., architecture, regions) goes in metadata.
    """

    name: str  # type_name (AWS) ou slug (DO, Verda)
    vcpu: int
    memory_gb: float
    accelerator: str | None = None
    accelerator_count: float = 0
    accelerator_memory_gb: float = 0
    price_on_demand: float | None = None  # USD per hour
    price_spot: float | None = None  # USD per hour (if available)
    billing_increment_minutes: int | None = None  # None = per-second billing
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_memory_mb(memory: Memory | None) -> int:
    """Parse memory to MB.

    Args:
        memory: Memory specification. Can be:
            - None or Auto(): Returns 1024 MB (1GB) default
            - int: Treated as MB directly
            - str: Parsed with suffix (e.g., '8GiB', '4096MB', '8G', '4096M')

    Returns:
        Memory in MB.
    """
    match memory:
        case None | Auto():
            return 1024
        case int() as mb:
            return mb
        case str() as s:
            s = s.strip().upper()
            if s.endswith("GIB"):
                return int(float(s[:-3]) * 1024)
            if s.endswith("MIB"):
                return int(float(s[:-3]))
            if s.endswith("GB"):
                return int(float(s[:-2]) * 1024)
            if s.endswith("MB"):
                return int(float(s[:-2]))
            if s.endswith("G"):
                return int(float(s[:-1]) * 1024)
            if s.endswith("M"):
                return int(float(s[:-1]))
            # Assume MB if no suffix
            return int(float(s))


def _normalize_accelerator_name(acc: str | AcceleratorSpec) -> str:
    """Normalize accelerator name for comparison (e.g., 'RTX 5090' -> 'RTX_5090')."""
    acc_str = acc.accelerator if isinstance(acc, AcceleratorSpec) else str(acc)
    return acc_str.upper().replace("-", "_").replace(" ", "_")


def _matches_count(spec_count: float, requirement: AcceleratorCount) -> bool:
    """Check if spec_count matches the requirement.

    Args:
        spec_count: Actual accelerator count from the instance spec.
        requirement: Either an exact number or a predicate function.

    Returns:
        True if spec_count matches the requirement.
    """
    if callable(requirement):
        return requirement(int(spec_count))
    return int(spec_count) == int(requirement)


def select_instance(
    instances: tuple[InstanceSpec, ...],
    cpu: int = 1,
    memory_mb: int = 1024,
    accelerator: str | AcceleratorSpec | None = None,
    accelerator_count: AcceleratorCount = 1,
    prefer_spot: bool = False,
) -> InstanceSpec:
    """Select cheapest instance that meets requirements.

    Generic selection function that works with any provider's instances.

    Args:
        instances: Available instances from provider.available_instances().
        cpu: Required number of vCPUs.
        memory_mb: Required memory in MB.
        accelerator: Required accelerator type (e.g., "H100", "A100").
        accelerator_count: Required number of accelerators. Can be:
            - int: Exact count (e.g., 2 means exactly 2 GPUs)
            - Callable[[int], bool]: Predicate function (e.g., lambda c: c >= 2)
        prefer_spot: If True, sort by spot price; otherwise by on-demand price.

    Returns:
        InstanceSpec that meets requirements (cheapest option).

    Raises:
        ValueError: If no instance type meets the requirements.
    """
    memory_gb = memory_mb / 1024

    def _price_key(s: InstanceSpec) -> float:
        if prefer_spot:
            return s.price_spot or s.price_on_demand or float("inf")
        return s.price_on_demand or float("inf")

    if accelerator:
        acc_normalized = _normalize_accelerator_name(accelerator)
        candidates = [
            s
            for s in instances
            if s.accelerator
            and _normalize_accelerator_name(s.accelerator).startswith(acc_normalized)
            and _matches_count(s.accelerator_count, accelerator_count)
            and s.vcpu >= cpu
            and s.memory_gb >= memory_gb
        ]

        if not candidates:
            available_counts = sorted(
                {
                    int(s.accelerator_count)
                    for s in instances
                    if s.accelerator
                    and _normalize_accelerator_name(s.accelerator).startswith(
                        acc_normalized
                    )
                }
            )
            if available_counts:
                raise NoMatchingInstanceError(
                    f"No instance type found for {accelerator_count}x {accelerator} "
                    f"with {cpu} vCPU, {memory_gb}GB RAM. "
                    f"Available counts for {accelerator}: {available_counts}"
                )
            available = sorted({s.accelerator for s in instances if s.accelerator})
            raise NoMatchingInstanceError(
                f"No instance type found for {accelerator}. "
                f"Available accelerator types: {available}"
            )

        candidates.sort(key=_price_key)
        return candidates[0]

    # Standard instances (no accelerator)
    candidates = [
        s
        for s in instances
        if not s.accelerator and s.vcpu >= cpu and s.memory_gb >= memory_gb
    ]

    if not candidates:
        # Fallback: include any instance that meets CPU/memory
        candidates = [
            s for s in instances if s.vcpu >= cpu and s.memory_gb >= memory_gb
        ]

    if not candidates:
        max_vcpu = max((s.vcpu for s in instances), default=0)
        max_mem = max((s.memory_gb for s in instances), default=0)
        raise NoMatchingInstanceError(
            f"No instance type found for {cpu} vCPU, {memory_gb}GB RAM. "
            f"Maximum available: {max_vcpu} vCPU, {max_mem}GB RAM"
        )

    candidates.sort(key=_price_key)
    return candidates[0]


def select_instances(
    instances: tuple[InstanceSpec, ...],
    cpu: int = 1,
    memory_mb: int = 1024,
    accelerator: str | AcceleratorSpec | None = None,
    accelerator_count: AcceleratorCount = 1,
    prefer_spot: bool = False,
) -> tuple[InstanceSpec, ...]:
    """Select all instances that meet requirements, sorted by price.

    Similar to select_instance but returns ALL matching candidates instead of
    just the cheapest. Useful for Fleet API which can try multiple instance
    types and pick one with available capacity.

    Args:
        instances: Available instances from provider.available_instances().
        cpu: Required number of vCPUs.
        memory_mb: Required memory in MB.
        accelerator: Required accelerator type (e.g., "H100", "A100").
        accelerator_count: Required number of accelerators. Can be:
            - int: Exact count (e.g., 2 means exactly 2 GPUs)
            - Callable[[int], bool]: Predicate function (e.g., lambda c: c >= 2)
        prefer_spot: If True, sort by spot price; otherwise by on-demand price.

    Returns:
        Tuple of InstanceSpecs that meet requirements, sorted by price (cheapest first).

    Raises:
        ValueError: If no instance type meets the requirements.
    """
    memory_gb = memory_mb / 1024

    def _price_key(s: InstanceSpec) -> float:
        if prefer_spot:
            return s.price_spot or s.price_on_demand or float("inf")
        return s.price_on_demand or float("inf")

    if accelerator:
        acc_normalized = _normalize_accelerator_name(accelerator)
        candidates = [
            s
            for s in instances
            if s.accelerator
            and _normalize_accelerator_name(s.accelerator).startswith(acc_normalized)
            and _matches_count(s.accelerator_count, accelerator_count)
            and s.vcpu >= cpu
            and s.memory_gb >= memory_gb
        ]

        if not candidates:
            available_counts = sorted(
                {
                    int(s.accelerator_count)
                    for s in instances
                    if s.accelerator
                    and _normalize_accelerator_name(s.accelerator).startswith(
                        acc_normalized
                    )
                }
            )
            if available_counts:
                raise NoMatchingInstanceError(
                    f"No instance type found for {accelerator_count}x {accelerator} "
                    f"with {cpu} vCPU, {memory_gb}GB RAM. "
                    f"Available counts for {accelerator}: {available_counts}"
                )
            available = sorted({s.accelerator for s in instances if s.accelerator})
            raise NoMatchingInstanceError(
                f"No instance type found for {accelerator}. "
                f"Available accelerator types: {available}"
            )

        candidates.sort(key=_price_key)
        return tuple(candidates)

    # Standard instances (no accelerator)
    candidates = [
        s
        for s in instances
        if not s.accelerator and s.vcpu >= cpu and s.memory_gb >= memory_gb
    ]

    if not candidates:
        # Fallback: include any instance that meets CPU/memory
        candidates = [
            s for s in instances if s.vcpu >= cpu and s.memory_gb >= memory_gb
        ]

    if not candidates:
        max_vcpu = max((s.vcpu for s in instances), default=0)
        max_mem = max((s.memory_gb for s in instances), default=0)
        raise NoMatchingInstanceError(
            f"No instance type found for {cpu} vCPU, {memory_gb}GB RAM. "
            f"Maximum available: {max_vcpu} vCPU, {max_mem}GB RAM"
        )

    candidates.sort(key=_price_key)
    return tuple(candidates)
