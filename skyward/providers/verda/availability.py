"""Instance availability checking for Verda.

Provides functions to query instance availability across regions
and automatically select a region where the requested instance type
is available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from verda import VerdaClient


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


def get_availability(
    client: VerdaClient,
    is_spot: bool = False,
) -> dict[str, frozenset[str]]:
    """Get instance availability across all regions.

    Args:
        client: Authenticated Verda client.
        is_spot: Check spot instance availability.

    Returns:
        Dict mapping region code to set of available instance types.
    """
    response = client._http_client.get(
        "/instance-availability",
        params={"is_spot": str(is_spot).lower()},
    )
    data = response.json()

    return {
        region["location_code"]: frozenset(region.get("availabilities", []))
        for region in data
    }


def find_available_region(
    client: VerdaClient,
    instance_type: str,
    is_spot: bool = False,
    preferred_region: str | None = None,
) -> str:
    """Find a region where the instance type is available.

    Checks availability across all regions and returns one where
    the requested instance type can be deployed. If a preferred
    region is specified and has availability, it will be returned.

    Args:
        client: Authenticated Verda client.
        instance_type: Instance type slug (e.g., "1L40S.20V").
        is_spot: Check spot instance availability.
        preferred_region: Try this region first if specified.

    Returns:
        Region code where instance is available.

    Raises:
        NoAvailableRegionError: If no region has availability.
    """
    availability = get_availability(client, is_spot)
    regions_checked = list(availability.keys())

    # Try preferred region first
    if preferred_region:
        available_types = availability.get(preferred_region, frozenset())
        if instance_type in available_types:
            return preferred_region

    # Find any region with availability
    for region, types in availability.items():
        if instance_type in types:
            return region

    raise NoAvailableRegionError(instance_type, is_spot, regions_checked)


def get_available_instance_types(
    client: VerdaClient,
    is_spot: bool = False,
    region: str | None = None,
) -> frozenset[str]:
    """Get all available instance types, optionally filtered by region.

    Args:
        client: Authenticated Verda client.
        is_spot: Check spot instance availability.
        region: Filter to specific region (None = all regions).

    Returns:
        Set of available instance type slugs.
    """
    availability = get_availability(client, is_spot)

    if region:
        return availability.get(region, frozenset())

    # Union of all regions
    all_types: set[str] = set()
    for types in availability.values():
        all_types.update(types)
    return frozenset(all_types)
