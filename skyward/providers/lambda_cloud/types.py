"""Lambda Cloud API response types.

TypedDicts for API responses — no conversion logic.
"""

from __future__ import annotations

import re
from typing import NotRequired, TypedDict


class RegionResponse(TypedDict):
    """Region from Lambda API."""

    name: str
    description: str


class InstanceSpecsResponse(TypedDict):
    """Instance hardware specs."""

    vcpus: int
    memory_gib: int
    storage_gib: int


class InstanceTypeInfo(TypedDict):
    """Instance type metadata."""

    name: str
    price_cents_per_hour: int
    description: str
    gpu_description: NotRequired[str]
    specs: InstanceSpecsResponse


class InstanceTypeEntry(TypedDict):
    """Instance type with regional availability."""

    instance_type: InstanceTypeInfo
    regions_with_capacity_available: list[RegionResponse]


class InstanceResponse(TypedDict):
    """Instance from Lambda API."""

    id: str
    name: NotRequired[str]
    ip: NotRequired[str]
    private_ip: NotRequired[str]
    status: str
    hostname: NotRequired[str]
    ssh_key_names: list[str]
    region: RegionResponse
    instance_type: InstanceTypeInfo
    is_reserved: NotRequired[bool]


class SSHKeyResponse(TypedDict):
    """SSH key from Lambda API."""

    id: str
    name: str
    public_key: str


class LaunchResponse(TypedDict):
    """Launch instances response."""

    instance_ids: list[str]


class ErrorDetail(TypedDict):
    """Error detail from Lambda API."""

    code: str
    message: str
    suggestion: NotRequired[str]


# =============================================================================
# Helper functions
# =============================================================================

_TYPE_NAME_RE = re.compile(r"gpu_(\d+)x_(.+)")


def parse_gpu_from_type_name(type_name: str) -> tuple[str, int]:
    """Parse GPU name and count from Lambda instance type name.

    Examples
    --------
    >>> parse_gpu_from_type_name("gpu_1x_a100")
    ('A100', 1)
    >>> parse_gpu_from_type_name("gpu_8x_h100_sxm5")
    ('H100 SXM5', 8)
    >>> parse_gpu_from_type_name("gpu_1x_a10")
    ('A10', 1)
    """
    match = _TYPE_NAME_RE.match(type_name)
    if not match:
        return (type_name.upper(), 1)
    count = int(match.group(1))
    gpu_part = match.group(2).upper().replace("_", " ")
    return (gpu_part, count)
