"""Instance specifications and selection functions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from skyward.accelerators import AcceleratorCount, AcceleratorSpec
from skyward.core.exceptions import BudgetExceededError, NoMatchingInstanceError
from skyward.types.core import Auto, Memory

__all__ = [
    "InstanceSpec",
    "InstanceStatus",
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


@dataclass(frozen=True, slots=True)
class InstanceStatus:
    """Current status of a cloud instance.

    Used by providers to report instance state for monitoring purposes.
    The `status` field contains provider-specific status strings
    (e.g., "running", "stopped", "outbid" for Vast.ai).
    """

    instance_id: str
    status: str
    ssh_available: bool = True


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


def _accelerator_matches(instance_acc: str, requested_acc: str) -> bool:
    """Check if instance accelerator matches the requested one.

    Handles:
    - Exact matches (case-insensitive)
    - Hyphen-separated variants (A100 matches A100-40, H100 matches H100-SXM)
    - Space-separated form factors (H100 NVL matches H100, A100 SXM4 matches A100)

    Does NOT match:
    - Different GPUs with similar prefixes (L4 vs L40, L40S)
    - Space-separated GPU variants (RTX 5070 vs RTX 5070 Ti are different GPUs)

    Examples:
        - "L4" matches "L4" ✓
        - "L4" matches "L40" ✗ (different GPU)
        - "A100-40" matches "A100" ✓ (hyphen = memory variant)
        - "H100-SXM" matches "H100" ✓ (hyphen = form factor)
        - "H100 NVL" matches "H100" ✓ (space + form factor suffix)
        - "RTX 5070 Ti" matches "RTX 5070" ✗ (different GPU)
    """
    # Normalize: uppercase, trim whitespace
    inst_norm = instance_acc.upper().strip()
    req_norm = requested_acc.upper().strip()

    # Exact match
    if inst_norm == req_norm:
        return True

    # Hyphen-separated variant: A100 matches A100-40, H100 matches H100-SXM
    if inst_norm.startswith(req_norm + "-"):
        return True

    # Space-separated form factor: H100 NVL matches H100, A100 SXM4 matches A100
    # Only match known form factor suffixes (NVL, SXM, SXM4, PCIe, PCIE)
    if inst_norm.startswith(req_norm + " "):
        suffix = inst_norm[len(req_norm) + 1:]
        form_factors = {"NVL", "SXM", "SXM4", "SXM5", "PCIE", "NVLINK"}
        if suffix in form_factors:
            return True

    return False


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
    max_price: float | None = None,
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
        max_price: Maximum price per hour (USD). None means no limit.

    Returns:
        InstanceSpec that meets requirements (cheapest option).

    Raises:
        NoMatchingInstanceError: If no instance type meets the requirements.
        BudgetExceededError: If no instance fits within the budget.
    """
    memory_gb = memory_mb / 1024

    def _price_key(s: InstanceSpec) -> float:
        if prefer_spot:
            return s.price_spot or s.price_on_demand or float("inf")
        return s.price_on_demand or float("inf")

    if accelerator:
        acc_name = accelerator.accelerator if isinstance(accelerator, AcceleratorSpec) else accelerator
        candidates = [
            s
            for s in instances
            if s.accelerator
            and _accelerator_matches(s.accelerator, acc_name)
            and _matches_count(s.accelerator_count, accelerator_count)
            and s.vcpu >= cpu
            and s.memory_gb >= memory_gb
        ]

        if not candidates:
            available_counts = sorted(
                {
                    int(s.accelerator_count)
                    for s in instances
                    if s.accelerator and _accelerator_matches(s.accelerator, acc_name)
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

        # Apply budget filter
        if max_price is not None:
            cheapest_price = _price_key(candidates[0])
            candidates = [c for c in candidates if _price_key(c) <= max_price]
            if not candidates:
                raise BudgetExceededError(
                    f"No instance found within budget ${max_price:.2f}/hr. "
                    f"Cheapest matching instance: ${cheapest_price:.2f}/hr"
                )

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

    # Apply budget filter
    if max_price is not None:
        cheapest_price = _price_key(candidates[0])
        candidates = [c for c in candidates if _price_key(c) <= max_price]
        if not candidates:
            raise BudgetExceededError(
                f"No instance found within budget ${max_price:.2f}/hr. "
                f"Cheapest matching instance: ${cheapest_price:.2f}/hr"
            )

    return candidates[0]


def select_instances(
    instances: tuple[InstanceSpec, ...],
    cpu: int = 1,
    memory_mb: int = 1024,
    accelerator: str | AcceleratorSpec | None = None,
    accelerator_count: AcceleratorCount = 1,
    prefer_spot: bool = False,
    max_price: float | None = None,
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
        max_price: Maximum price per hour (USD). None means no limit.

    Returns:
        Tuple of InstanceSpecs that meet requirements, sorted by price (cheapest first).

    Raises:
        NoMatchingInstanceError: If no instance type meets the requirements.
        BudgetExceededError: If no instance fits within the budget.
    """
    memory_gb = memory_mb / 1024

    def _price_key(s: InstanceSpec) -> float:
        if prefer_spot:
            return s.price_spot or s.price_on_demand or float("inf")
        return s.price_on_demand or float("inf")

    if accelerator:
        acc_name = accelerator.accelerator if isinstance(accelerator, AcceleratorSpec) else accelerator
        candidates = [
            s
            for s in instances
            if s.accelerator
            and _accelerator_matches(s.accelerator, acc_name)
            and _matches_count(s.accelerator_count, accelerator_count)
            and s.vcpu >= cpu
            and s.memory_gb >= memory_gb
        ]

        if not candidates:
            available_counts = sorted(
                {
                    int(s.accelerator_count)
                    for s in instances
                    if s.accelerator and _accelerator_matches(s.accelerator, acc_name)
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

        # Apply budget filter
        if max_price is not None:
            cheapest_price = _price_key(candidates[0])
            candidates = [c for c in candidates if _price_key(c) <= max_price]
            if not candidates:
                raise BudgetExceededError(
                    f"No instance found within budget ${max_price:.2f}/hr. "
                    f"Cheapest matching instance: ${cheapest_price:.2f}/hr"
                )

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

    # Apply budget filter
    if max_price is not None:
        cheapest_price = _price_key(candidates[0])
        candidates = [c for c in candidates if _price_key(c) <= max_price]
        if not candidates:
            raise BudgetExceededError(
                f"No instance found within budget ${max_price:.2f}/hr. "
                f"Cheapest matching instance: ${cheapest_price:.2f}/hr"
            )

    return tuple(candidates)
