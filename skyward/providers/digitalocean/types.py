"""TypedDicts for DigitalOcean API responses.

These match the DigitalOcean API v2 response structure.
"""

from __future__ import annotations

from typing import TypedDict


# =============================================================================
# SSH Key Types
# =============================================================================


class SSHKeyResponse(TypedDict):
    """SSH key from DigitalOcean API."""

    id: int
    fingerprint: str
    public_key: str
    name: str


class SSHKeysListResponse(TypedDict):
    """Response from list SSH keys endpoint."""

    ssh_keys: list[SSHKeyResponse]


class CreateSSHKeyResponse(TypedDict):
    """Response from create SSH key endpoint."""

    ssh_key: SSHKeyResponse


# =============================================================================
# Size (Instance Type) Types
# =============================================================================


class GPUVRAMInfo(TypedDict, total=False):
    """GPU VRAM information."""

    amount: int  # GB
    unit: str


class GPUInfo(TypedDict, total=False):
    """GPU information for a size."""

    model: str  # e.g., "nvidia_h100"
    count: int
    vram: GPUVRAMInfo


class SizeResponse(TypedDict, total=False):
    """Droplet size from DigitalOcean API."""

    slug: str
    memory: int  # MB
    vcpus: int
    disk: int  # GB
    transfer: float  # TB
    price_monthly: float
    price_hourly: float
    regions: list[str]
    available: bool
    gpu_info: GPUInfo | None


class SizesListResponse(TypedDict):
    """Response from list sizes endpoint."""

    sizes: list[SizeResponse]


# =============================================================================
# Droplet Types
# =============================================================================


class NetworkV4(TypedDict, total=False):
    """IPv4 network interface."""

    ip_address: str
    netmask: str
    gateway: str
    type: str  # "public" or "private"


class NetworkV6(TypedDict, total=False):
    """IPv6 network interface."""

    ip_address: str
    netmask: int
    gateway: str
    type: str


class Networks(TypedDict, total=False):
    """Network configuration."""

    v4: list[NetworkV4]
    v6: list[NetworkV6]


class DropletResponse(TypedDict, total=False):
    """Droplet from DigitalOcean API."""

    id: int
    name: str
    memory: int
    vcpus: int
    disk: int
    locked: bool
    status: str  # "new", "active", "off", "archive"
    created_at: str
    networks: Networks
    region: dict  # Contains slug, name, etc.
    image: dict
    size: dict
    size_slug: str
    tags: list[str]


class CreateDropletResponse(TypedDict):
    """Response from create droplet endpoint."""

    droplet: DropletResponse


class GetDropletResponse(TypedDict):
    """Response from get droplet endpoint."""

    droplet: DropletResponse


class DropletsListResponse(TypedDict):
    """Response from list droplets endpoint."""

    droplets: list[DropletResponse]


# =============================================================================
# Helper Functions
# =============================================================================


def get_public_ip(droplet: DropletResponse) -> str | None:
    """Extract public IPv4 address from droplet."""
    networks = droplet.get("networks", {})
    for network in networks.get("v4", []):
        if network.get("type") == "public":
            return network.get("ip_address")
    return None


def get_private_ip(droplet: DropletResponse) -> str | None:
    """Extract private IPv4 address from droplet."""
    networks = droplet.get("networks", {})
    for network in networks.get("v4", []):
        if network.get("type") == "private":
            return network.get("ip_address")
    return None


def normalize_gpu_model(model: str | None) -> str | None:
    """Normalize DigitalOcean GPU model name.

    Args:
        model: Raw GPU model string from API (e.g., "nvidia_h100").

    Returns:
        Normalized model name (e.g., "H100") or None.
    """
    if not model:
        return None

    model = model.lower()
    for prefix in ("nvidia_", "amd_"):
        if model.startswith(prefix):
            model = model[len(prefix) :]
            break

    return model.upper()


def get_gpu_image(accelerator: str | None, accelerator_count: int = 1) -> str:
    """Get the appropriate GPU image for the accelerator type.

    Args:
        accelerator: Accelerator name (e.g., "H100").
        accelerator_count: Number of GPUs.

    Returns:
        DigitalOcean image slug.
    """
    if accelerator and "MI3" in accelerator.upper():
        return "gpu-amd-base"

    if accelerator_count == 8:
        return "gpu-h100x8-base"

    return "gpu-h100x1-base"


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # SSH Keys
    "SSHKeyResponse",
    "SSHKeysListResponse",
    "CreateSSHKeyResponse",
    # Sizes
    "GPUInfo",
    "GPUVRAMInfo",
    "SizeResponse",
    "SizesListResponse",
    # Droplets
    "NetworkV4",
    "NetworkV6",
    "Networks",
    "DropletResponse",
    "CreateDropletResponse",
    "GetDropletResponse",
    "DropletsListResponse",
    # Helpers
    "get_public_ip",
    "get_private_ip",
    "normalize_gpu_model",
    "get_gpu_image",
]
