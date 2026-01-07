"""Spec module - environment specifications.

Contains allocation specs, image definitions, and volume management.
"""

from skyward.spec.allocation import (
    Allocation,
    AllocationLike,
    AllocationLiteral,
    AllocationStrategy,
    NormalizedAllocation,
    normalize_allocation,
)
from skyward.spec.image import (
    DEFAULT_IMAGE,
    Image,
    SkywardSource,
)
from skyward.spec.volume import (
    S3Volume,
    Volume,
    parse_volume_uri,
)

__all__ = [
    # Allocation
    "Allocation",
    "AllocationLike",
    "AllocationLiteral",
    "AllocationStrategy",
    "NormalizedAllocation",
    "normalize_allocation",
    # Image
    "Image",
    "DEFAULT_IMAGE",
    "SkywardSource",
    # Volume
    "Volume",
    "S3Volume",
    "parse_volume_uri",
]
