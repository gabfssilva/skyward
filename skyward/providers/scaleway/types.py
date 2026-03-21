"""Scaleway API response types.

TypedDicts for the Instance API and IAM API responses.
"""

from __future__ import annotations

from typing import NotRequired, TypedDict


class ServerTypeGpuInfo(TypedDict):
    """GPU info nested in a ServerType."""

    gpu_manufacturer: str
    gpu_name: str
    gpu_memory: int


class ServerTypeResponse(TypedDict):
    """Server type from list_servers_types.

    The endpoint returns {"servers": {"TYPE_NAME": {...}, ...}}.
    """

    hourly_price: float
    ncpus: int
    ram: int
    arch: str
    gpu: NotRequired[int]
    gpu_info: NotRequired[ServerTypeGpuInfo]
    monthly_price: NotRequired[float]
    per_volume_constraint: NotRequired[dict]
    volumes_constraint: NotRequired[dict]


class ServerIpResponse(TypedDict):
    """IP address info nested in a server."""

    id: str
    address: str
    dynamic: bool
    family: NotRequired[str]
    gateway: NotRequired[str]
    netmask: NotRequired[str]
    provisioning_mode: NotRequired[str]
    state: NotRequired[str]


class ServerVolumeResponse(TypedDict):
    """Volume info nested in a server response."""

    id: str
    name: NotRequired[str]
    volume_type: NotRequired[str]
    size: NotRequired[int]
    state: NotRequired[str]
    boot: NotRequired[bool]


class ServerResponse(TypedDict):
    """Server from the Instance API."""

    id: str
    name: str
    state: str
    commercial_type: str
    public_ip: NotRequired[ServerIpResponse]
    public_ips: NotRequired[list[ServerIpResponse]]
    private_ip: NotRequired[str]
    image: NotRequired[ServerImageResponse]
    volumes: NotRequired[dict[str, ServerVolumeResponse]]
    arch: NotRequired[str]
    project: NotRequired[str]
    tags: NotRequired[list[str]]
    creation_date: NotRequired[str]
    modification_date: NotRequired[str]


class ServerImageResponse(TypedDict):
    """Image info nested in a server."""

    id: str
    name: NotRequired[str]


class CreateServerResponse(TypedDict):
    """Response from server creation."""

    server: ServerResponse


class IpResponse(TypedDict):
    """Flexible IP from the IP API."""

    id: str
    address: str
    server: NotRequired[ServerResponse | None]
    project: NotRequired[str]
    zone: NotRequired[str]
    type: NotRequired[str]
    state: NotRequired[str]


class SSHKeyResponse(TypedDict):
    """SSH key from the IAM API."""

    id: str
    name: str
    public_key: str
    fingerprint: NotRequired[str]
    organization_id: NotRequired[str]
    project_id: NotRequired[str]
    disabled: NotRequired[bool]
    created_at: NotRequired[str]
    updated_at: NotRequired[str]


class ImageResponse(TypedDict):
    """OS image from the Instance API."""

    id: str
    name: str
    arch: NotRequired[str]
    creation_date: NotRequired[str]
    public: NotRequired[bool]
    from_server: NotRequired[str]
    organization: NotRequired[str]
    project: NotRequired[str]
    tags: NotRequired[list[str]]
    state: NotRequired[str]


# ============================================================================
# GPU name helpers
# ============================================================================

_GPU_SUFFIXES = ("-SXM", "-PCIe", "-80G", "-48G", "-24G", "-288G")


def normalize_gpu_name(name: str) -> str:
    """Strip form-factor and VRAM suffixes for matching.

    Examples
    --------
    >>> normalize_gpu_name("H100 SXM")
    'H100'
    >>> normalize_gpu_name("NVIDIA L40S")
    'L40S'
    """
    upper = name.upper().replace("NVIDIA ", "").strip()
    for suffix in _GPU_SUFFIXES:
        upper = upper.replace(suffix.upper(), "")
    return upper.strip("-").replace("-", "_").replace(" ", "_")


# Mapping from Scaleway commercial_type prefix to (gpu_model, gpu_count, vram_gb)
COMMERCIAL_TYPE_GPU_MAP: dict[str, tuple[str, int, int]] = {
    "L4-1-24G": ("L4", 1, 24),
    "L4-2-24G": ("L4", 2, 24),
    "L4-4-24G": ("L4", 4, 24),
    "L4-8-24G": ("L4", 8, 24),
    "L40S-1-48G": ("L40S", 1, 48),
    "L40S-2-48G": ("L40S", 2, 48),
    "L40S-4-48G": ("L40S", 4, 48),
    "L40S-8-48G": ("L40S", 8, 48),
    "H100-1-80G": ("H100", 1, 80),
    "H100-2-80G": ("H100", 2, 80),
    "H100-SXM-2-80G": ("H100 SXM", 2, 80),
    "H100-SXM-4-80G": ("H100 SXM", 4, 80),
    "H100-SXM-8-80G": ("H100 SXM", 8, 80),
    "B300-SXM-8-288G": ("B300", 8, 288),
    "RENDER-S": ("Tesla P100", 1, 16),
}
