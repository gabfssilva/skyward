"""Vast.ai API response types.

TypedDicts for API responses - no conversion needed.
"""

from __future__ import annotations

from typing import NotRequired, TypedDict


class OfferResponse(TypedDict):
    """GPU offer from the marketplace."""

    id: int
    machine_id: int
    gpu_name: str
    num_gpus: int
    cpu_cores: int
    cpu_ram: float  # MB (API returns MB)
    gpu_ram: float  # MB
    disk_space: float  # GB
    reliability: float
    dph_total: float  # on-demand $/hr
    min_bid: float  # spot $/hr
    cuda_max_good: float
    geolocation: str
    cluster_id: int | None
    inet_up: float  # Mbps
    inet_down: float  # Mbps
    dlperf: float | None
    verified: bool


class InstanceResponse(TypedDict):
    """Instance status from API."""

    id: int
    actual_status: str
    ssh_host: str
    ssh_port: int
    gpu_name: str
    num_gpus: int
    machine_id: int
    label: str | None
    is_bid: bool
    dph_total: float
    public_ipaddr: str
    ports: NotRequired[dict[str, list[dict[str, int]]]]  # e.g. {"22/tcp": [{"HostPort": 12345}]}


class SSHKeyResponse(TypedDict):
    """SSH key from API."""

    id: int
    public_key: str


class OverlayResponse(TypedDict):
    """Overlay network from API."""

    overlay_id: NotRequired[int]
    id: NotRequired[int]
    name: str
    cluster_id: int
    instances: NotRequired[list[int]]


class BundlesResponse(TypedDict):
    """Bundles/offers search response."""

    offers: list[OfferResponse]


class SSHKeysListResponse(TypedDict):
    """SSH keys list response."""

    ssh_keys: list[SSHKeyResponse]


class InstancesListResponse(TypedDict):
    """Instances list response."""

    instances: list[InstanceResponse]


class InstanceGetResponse(TypedDict):
    """Single instance get response."""

    instances: InstanceResponse


class CreateInstanceResponse(TypedDict):
    """Create instance response."""

    new_contract: NotRequired[int]
    id: NotRequired[int]
    success: NotRequired[bool]


class CreateSSHKeyResponse(TypedDict):
    """Create SSH key response."""

    ssh_key_id: NotRequired[int]
    id: NotRequired[int]
    success: NotRequired[bool]
    error: NotRequired[str]


class OverlayCreateResponse(TypedDict):
    """Overlay creation response."""

    success: bool
    msg: NotRequired[str]
    error: NotRequired[str]


# =============================================================================
# Helper functions for accessing typed data
# =============================================================================


def get_direct_ssh_port(instance: InstanceResponse) -> int | None:
    """Extract direct SSH port from ports mapping."""
    ports = instance.get("ports")
    if not ports:
        return None
    ssh_mapping = ports.get("22/tcp")
    if ssh_mapping and len(ssh_mapping) > 0:
        return ssh_mapping[0].get("HostPort")
    return None


def cpu_ram_gb(offer: OfferResponse) -> float:
    """Get CPU RAM in GB (API returns MB)."""
    return offer["cpu_ram"] / 1024


def gpu_ram_gb(offer: OfferResponse) -> float:
    """Get GPU RAM in GB (API returns MB)."""
    return offer["gpu_ram"] / 1024 if offer["gpu_ram"] else 0


def normalized_gpu_name(offer: OfferResponse) -> str:
    """Normalized GPU name (e.g., 'RTX 5090' -> 'RTX_5090')."""
    name = offer["gpu_name"].upper()
    for suffix in ("_PCIE", "_SXM", "_80GB", "_40GB"):
        name = name.replace(suffix, "")
    return name.replace(" ", "_")


__all__ = [
    # Response types
    "OfferResponse",
    "InstanceResponse",
    "SSHKeyResponse",
    "OverlayResponse",
    "BundlesResponse",
    "SSHKeysListResponse",
    "InstancesListResponse",
    "InstanceGetResponse",
    "CreateInstanceResponse",
    "CreateSSHKeyResponse",
    "OverlayCreateResponse",
    # Helpers
    "get_direct_ssh_port",
    "cpu_ram_gb",
    "gpu_ram_gb",
    "normalized_gpu_name",
]
