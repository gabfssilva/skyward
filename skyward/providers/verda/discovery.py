"""Verda instance type discovery, parsing, and availability checking."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from skyward.cache import cached
from skyward.types import InstanceSpec

if TYPE_CHECKING:
    from verda import VerdaClient


# =============================================================================
# Constants
# =============================================================================

GPU_MODEL_NORMALIZATIONS = {
    "Tesla V100": "V100",
    "RTX A6000": "A6000",
    "RTX 6000 Ada": "RTX6000Ada",
    "RTX PRO 6000": "RTXPRO6000",
}


# =============================================================================
# Exceptions
# =============================================================================


class NoAvailableRegionError(Exception):
    """No region has the requested instance type available."""

    def __init__(self, instance_type: str, is_spot: bool, regions_checked: list[str]):
        spot_str = "spot " if is_spot else ""
        regions_str = ", ".join(regions_checked) if regions_checked else "none"
        super().__init__(
            f"No region has {spot_str}instance type '{instance_type}' available. "
            f"Regions checked: {regions_str}. "
            "Try a different instance type or check Verda's capacity status."
        )
        self.instance_type = instance_type
        self.is_spot = is_spot
        self.regions_checked = regions_checked


# =============================================================================
# Instance Type Parsing
# =============================================================================


def parse_gpu_model(description: str) -> str | None:
    """Parse GPU model from Verda description string.

    Args:
        description: GPU description (e.g., "8x H100 SXM5 80GB")

    Returns:
        Normalized GPU model (e.g., "H100") or None.
    """
    if not description:
        return None

    match = re.match(r"^\d+x\s+(.+?)\s+\d+GB$", description)
    if not match:
        return None

    model_raw = match.group(1)
    model_clean = re.sub(r"\s+SXM\d+", "", model_raw)

    if model_clean in GPU_MODEL_NORMALIZATIONS:
        return GPU_MODEL_NORMALIZATIONS[model_clean]

    return model_clean.replace(" ", "")


def parse_instance_type(spec: dict[str, Any]) -> InstanceSpec:
    """Parse Verda raw API instance type to InstanceSpec.

    Args:
        spec: Raw instance type dict from Verda API.

    Returns:
        InstanceSpec with normalized fields.
    """
    cpu_info = spec.get("cpu") or {}
    vcpu = cpu_info.get("number_of_cores", 0) if isinstance(cpu_info, dict) else 0

    memory_info = spec.get("memory") or {}
    memory_gb = memory_info.get("size_in_gigabytes", 0) if isinstance(memory_info, dict) else 0

    gpu_info = spec.get("gpu") or {}
    accelerator = None
    accelerator_count = 0
    accelerator_memory_gb = 0.0

    if gpu_info and isinstance(gpu_info, dict):
        description = gpu_info.get("description", "")
        accelerator = parse_gpu_model(description)
        accelerator_count = gpu_info.get("number_of_gpus", 0)

        gpu_memory = spec.get("gpu_memory") or {}
        if isinstance(gpu_memory, dict):
            accelerator_memory_gb = gpu_memory.get("size_in_gigabytes", 0)

    # Extract pricing (API returns strings, convert to float)
    def _safe_float(v: Any) -> float | None:
        if v is None:
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    price_on_demand = _safe_float(spec.get("price_per_hour"))
    price_spot = _safe_float(spec.get("spot_price"))

    storage_info = spec.get("storage") or {}
    metadata = {
        "instance_id": spec.get("id"),
        "description": spec.get("description"),
        "storage": storage_info.get("description") if isinstance(storage_info, dict) else None,
        "supported_os": sorted(spec.get("supported_os") or [], reverse=True),
    }
    metadata = {k: v for k, v in metadata.items() if v is not None}

    return InstanceSpec(
        name=spec["instance_type"],
        vcpu=vcpu,
        memory_gb=memory_gb,
        accelerator=accelerator,
        accelerator_count=accelerator_count,
        accelerator_memory_gb=accelerator_memory_gb,
        price_on_demand=price_on_demand,
        price_spot=price_spot,
        billing_increment_minutes=10,  # Verda charges per 10-minute block
        metadata=metadata,
    )


# =============================================================================
# Availability & Region Discovery
# =============================================================================


def get_availability(client: VerdaClient, is_spot: bool = False) -> dict[str, frozenset[str]]:
    """Get instance availability across all regions.

    Args:
        client: Authenticated Verda client.
        is_spot: Check spot availability instead of on-demand.

    Returns:
        Dict mapping region code to set of available instance types.
    """
    response = client._http_client.get(
        "/instance-availability",
        params={"is_spot": str(is_spot).lower()},
    )
    data = response.json()
    return {region["location_code"]: frozenset(region.get("availabilities", [])) for region in data}


def find_available_region(
    client: VerdaClient,
    instance_type: str,
    is_spot: bool = False,
    preferred_region: str | None = None,
) -> str:
    """Find a region where the instance type is available.

    Checks the preferred region first, then falls back to any available region.

    Args:
        client: Authenticated Verda client.
        instance_type: Instance type to find.
        is_spot: Check spot availability.
        preferred_region: Preferred region to check first.

    Returns:
        Region code where instance type is available.

    Raises:
        NoAvailableRegionError: If no region has the instance type.
    """
    availability = get_availability(client, is_spot)
    regions_checked = list(availability.keys())

    if preferred_region:
        available_types = availability.get(preferred_region, frozenset())
        if instance_type in available_types:
            return preferred_region

    for region, types in availability.items():
        if instance_type in types:
            return region

    raise NoAvailableRegionError(instance_type, is_spot, regions_checked)


# =============================================================================
# Instance Type Fetching
# =============================================================================


@cached(namespace="verda")
def fetch_available_instances(client: VerdaClient) -> tuple[InstanceSpec, ...]:
    """Fetch all available instance types from Verda API.

    Results are cached for 24 hours.

    Args:
        client: Authenticated Verda client.

    Returns:
        Tuple of InstanceSpec sorted by accelerator, count, vcpu, memory.
    """
    # Use raw API to get supported_os field
    response = client._http_client.get("/instance-types")
    instance_types: list[dict[str, Any]] = response.json()

    return tuple(
        sorted(
            [parse_instance_type(t) for t in instance_types],
            key=lambda s: (
                s.accelerator or "",
                s.accelerator_count,
                s.vcpu,
                s.memory_gb,
            ),
        )
    )
