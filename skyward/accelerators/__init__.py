"""Accelerator specifications and types.

Public API:
    - Accelerator: Factory function for creating AcceleratorSpec
    - AcceleratorSpec: Immutable accelerator specification dataclass
    - AcceleratorCount: Type alias for count (int or predicate)
    - current_accelerator: Get accelerator for current compute pool
    - NVIDIA, AMD, TPU, etc.: Literal types for autocomplete
"""

from .spec import Accelerator, AcceleratorCount, AcceleratorSpec, current_accelerator
from .types import AMD, GPU, NVIDIA, TPU, Habana, Inferentia, Trainium

__all__ = [
    # Factory + dataclass
    "Accelerator",
    "AcceleratorSpec",
    "AcceleratorCount",
    "current_accelerator",
    # Literal types
    "NVIDIA",
    "AMD",
    "TPU",
    "Trainium",
    "Inferentia",
    "Habana",
    "GPU",
]
