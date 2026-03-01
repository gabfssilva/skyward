"""Hyperstack API response types.

TypedDicts for API responses and helper functions for GPU name normalization.
"""

from __future__ import annotations

from typing import NotRequired, TypedDict


class FlavorResponse(TypedDict):
    """Hardware flavor from Hyperstack API."""

    id: int
    name: str
    cpu: int
    ram: float
    disk: int
    gpu: str
    gpu_count: int
    region_name: NotRequired[str]
    stock_available: NotRequired[bool]


class ImageResponse(TypedDict):
    """OS image from Hyperstack API."""

    id: int
    name: str
    region_name: NotRequired[str]
    type: NotRequired[str]


class EnvironmentResponse(TypedDict):
    """Environment grouping resources."""

    id: int
    name: str
    region: NotRequired[str]


class KeypairResponse(TypedDict):
    """SSH keypair scoped to an environment."""

    id: int
    name: str
    fingerprint: NotRequired[str]
    public_key: NotRequired[str]
    environment: NotRequired[dict]


class VMResponse(TypedDict):
    """Virtual machine from Hyperstack API."""

    id: int
    name: NotRequired[str]
    status: str
    fixed_ip: NotRequired[str]
    floating_ip: NotRequired[str]
    floating_ip_status: NotRequired[str]
    flavor: NotRequired[FlavorInVM]
    image: NotRequired[ImageInVM]
    environment: NotRequired[dict]
    created_at: NotRequired[str]


class FlavorInVM(TypedDict):
    """Flavor info nested within a VM response."""

    name: NotRequired[str]
    cpu: NotRequired[int]
    ram: NotRequired[float]
    gpu: NotRequired[str]
    gpu_count: NotRequired[int]


class ImageInVM(TypedDict):
    """Image info nested within a VM response."""

    name: NotRequired[str]


class CreateVMPayload(TypedDict, total=False):
    """Payload for creating VMs."""

    name: str
    environment_name: str
    image_name: str
    flavor_name: str
    key_name: str
    assign_floating_ip: bool
    user_data: str
    count: int


class CreateVMResponse(TypedDict):
    """Response from VM creation."""

    status: NotRequired[bool]
    message: NotRequired[str]
    instances: NotRequired[list[VMResponse]]


class PricebookEntry(TypedDict):
    """Pricing entry from the pricebook.

    The /pricebook endpoint returns a flat list of these entries.
    ``name`` is the resource/GPU name, ``value`` is the hourly price.
    """

    id: NotRequired[int]
    name: NotRequired[str]
    value: NotRequired[float]
    original_value: NotRequired[float]
    discount_applied: NotRequired[bool]


# =============================================================================
# Helper functions
# =============================================================================


_GPU_SUFFIXES = ("-PCIe", "-SXM", "-SXM4", "-80G", "-40G", "-80GB", "-40GB")


def normalize_gpu_name(gpu: str) -> str:
    """Strip model/form-factor suffixes for matching.

    Examples
    --------
    >>> normalize_gpu_name("A100-80G-PCIe")
    'A100'
    >>> normalize_gpu_name("RTX-4090")
    'RTX_4090'
    """
    upper = gpu.upper()
    for suffix in _GPU_SUFFIXES:
        upper = upper.replace(suffix.upper(), "")
    return upper.strip("-").replace("-", "_").replace(" ", "_")


def flavor_gpu_count(flavor: FlavorResponse) -> int:
    """Extract GPU count from a flavor."""
    return flavor.get("gpu_count", 0)
