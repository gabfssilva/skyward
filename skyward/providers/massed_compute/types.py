"""Massed Compute API response types.

TypedDicts for API responses --- no conversion logic.
"""

from __future__ import annotations

import re
from typing import NotRequired, TypedDict


class InstanceTypeSpecs(TypedDict):
    vcpu_count: int
    memory_gib: int
    storage_gb: int


class InstanceTypeResponse(TypedDict):
    name: str
    description: str
    price_cents_per_hour: int
    specs: InstanceTypeSpecs


class RegionResponse(TypedDict):
    name: str
    description: str


class InventoryItem(TypedDict):
    instance_type: InstanceTypeResponse
    regions_with_capacity_available: list[RegionResponse]
    capacity_available: int


class ImageResponse(TypedDict):
    vm_image_id: int
    vm_image_name: str
    vm_image_description: NotRequired[str]


class ProductInfo(TypedDict):
    name: str
    description: str
    gpu_count: int
    vcpu: int
    ram: int
    storage: int
    price_hr: str
    final_price_hr: str


class RegionInfo(TypedDict):
    name: str
    description: str


class InstanceResponse(TypedDict):
    uuid: str
    name: NotRequired[str]
    ip: str
    username: str
    password: str
    status: str
    os_booted: int
    command_startup: NotRequired[str | None]
    created: NotRequired[str]
    active: NotRequired[int]
    region: NotRequired[RegionInfo]
    image: NotRequired[dict[str, str | int]]
    product: NotRequired[ProductInfo]


class SSHKeyResponse(TypedDict):
    id: str
    name: str
    public_key: NotRequired[str]


_GPU_PATTERN = re.compile(
    r"^(?:gpu_)?(\d+)x_(.+?)(?:_(?:spot|low_ram|high_ram|nvlink))?\s*$",
    re.IGNORECASE,
)

_GPU_NAME_MAP: dict[str, str] = {
    "a30": "A30",
    "a5000": "RTX A5000",
    "a6000": "RTX A6000",
    "a6000_low_ram": "RTX A6000",
    "a6000_high_ram": "RTX A6000",
    "a6000_nvlink": "RTX A6000",
    "6000_ada": "RTX 6000 Ada",
    "a100": "A100",
    "a100_sxm4": "A100",
    "dgx_a100": "A100",
    "h100": "H100",
    "h100_sxm5": "H100",
    "h100_nvl": "H100-NVL",
    "h100_nvl_nvlink": "H100-NVL",
    "h200_nvl": "H200-NVL",
    "h200_nvl_nvlink": "H200-NVL",
    "l40": "L40",
    "l40s": "L40S",
    "pro_6000_blackwell": "RTX PRO 6000",
}


def parse_product_name(product_name: str) -> tuple[int, str]:
    """Extract GPU count and catalog name from a Massed Compute product name.

    Parameters
    ----------
    product_name
        Product name like ``"gpu_1x_a6000"`` or ``"gpu_8x_h100_spot"``.

    Returns
    -------
    tuple[int, str]
        GPU count and normalized accelerator catalog name.
    """
    match = _GPU_PATTERN.match(product_name)
    if not match:
        return 1, product_name

    count = int(match.group(1))
    raw_gpu = match.group(2).lower()

    if catalog_name := _GPU_NAME_MAP.get(raw_gpu):
        return count, catalog_name

    return count, raw_gpu.upper()


def is_spot_product(product_name: str) -> bool:
    """Check if a product name denotes a spot instance."""
    return "_spot" in product_name.lower()
