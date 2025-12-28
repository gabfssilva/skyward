"""Accelerator type definitions and detection utilities.

Provides type-safe Literal types for supported accelerators and
utility functions for accelerator detection and classification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal, Union

if TYPE_CHECKING:
    from skyward.cluster import InstanceInfo


# =============================================================================
# NVIDIA GPUs
# =============================================================================

NVIDIA = Literal[
    # Legacy
    "K80",
    "P4",
    "P100",
    "V100",
    # Mid-range
    "T4",
    "A2",
    "A10",
    "A10G",
    # Enterprise (Ada/Lovelace)
    "L4",
    "L40",
    "L40S",
    # Ampere
    "A100-40",
    "A100-80",
    "A100-40GB",
    "A100-80GB",
    # Hopper
    "H100-80GB",
    "H100-NVL",
    "H100-SXM",
    "H100-PCIe",
    "H200",
    # Blackwell
    "B100",
    "B200",
    "GB200",
    # Grace Hopper
    "GH200",
    # Consumer GPUs (independent clouds)
    "RTX3080",
    "RTX3090",
    "RTX4080",
    "RTX4090",
]


# =============================================================================
# AWS Trainium / Inferentia
# =============================================================================

Trainium = Literal[
    "Trainium1",
    "Trainium2",
    "Trainium3",
]

Inferentia = Literal[
    "Inferentia1",
    "Inferentia2",
]


# =============================================================================
# Google TPUs
# =============================================================================

TPU = Literal[
    "TPUv2",
    "TPUv3",
    "TPUv4",
    "TPUv5e",
    "TPUv5p",
    "TPUv6",
    # TPU slices
    "TPUv2-8",
    "TPUv3-8",
    "TPUv3-32",
    "TPUv4-64",
    "TPUv5e-4",
    "TPUv5p-8",
]


# =============================================================================
# AMD GPUs
# =============================================================================

AMD = Literal[
    "MI50",
    "MI100",
    "MI210",
    "MI250",
    "MI250X",
    "MI300A",
    "MI300B",
    "MI300X",
]


# =============================================================================
# Habana (Gaudi)
# =============================================================================

Habana = Literal[
    "Gaudi",
    "Gaudi2",
    "Gaudi3",
]


# =============================================================================
# Union Types
# =============================================================================

GPU = Union[
    NVIDIA,
    AMD,
    None,
]

Accelerator = Union[
    NVIDIA,
    Trainium,
    Inferentia,
    TPU,
    AMD,
    Habana,
    None,
]


# =============================================================================
# Explicit Value Sets (for runtime type checking)
# =============================================================================

_NVIDIA_VALUES: Final[frozenset[str]] = frozenset({
    "K80", "P4", "P100", "V100",
    "T4", "A2", "A10", "A10G",
    "L4", "L40", "L40S",
    "A100-40", "A100-80",
    "A100-40GB", "A100-80GB",
    "H100-80GB", "H100-NVL", "H100-SXM", "H100-PCIe", "H200",
    "B100", "B200", "GB200",
    "GH200",
    "RTX3080", "RTX3090", "RTX4080", "RTX4090",
})

_TRAINIUM_VALUES: Final[frozenset[str]] = frozenset({
    "Trainium1", "Trainium2", "Trainium3",
})

_INFERENTIA_VALUES: Final[frozenset[str]] = frozenset({
    "Inferentia1", "Inferentia2",
})

_TPU_VALUES: Final[frozenset[str]] = frozenset({
    "TPUv2", "TPUv3", "TPUv4", "TPUv5e", "TPUv5p", "TPUv6",
    "TPUv2-8", "TPUv3-8", "TPUv3-32", "TPUv4-64", "TPUv5e-4", "TPUv5p-8",
})

_AMD_VALUES: Final[frozenset[str]] = frozenset({
    "MI50", "MI100", "MI210", "MI250", "MI250X", "MI300A", "MI300B", "MI300X",
})

_HABANA_VALUES: Final[frozenset[str]] = frozenset({
    "Gaudi", "Gaudi2", "Gaudi3",
})


# =============================================================================
# Type Checking Functions
# =============================================================================


def is_nvidia(acc: Accelerator) -> bool:
    """Check if accelerator is an NVIDIA GPU."""
    return acc in _NVIDIA_VALUES


def is_trainium(acc: Accelerator) -> bool:
    """Check if accelerator is AWS Trainium."""
    return acc in _TRAINIUM_VALUES


def is_inferentia(acc: Accelerator) -> bool:
    """Check if accelerator is AWS Inferentia."""
    return acc in _INFERENTIA_VALUES


def is_tpu(acc: Accelerator) -> bool:
    """Check if accelerator is a Google TPU."""
    return acc in _TPU_VALUES


def is_amd(acc: Accelerator) -> bool:
    """Check if accelerator is an AMD GPU."""
    return acc in _AMD_VALUES


def is_habana(acc: Accelerator) -> bool:
    """Check if accelerator is a Habana Gaudi."""
    return acc in _HABANA_VALUES


def is_gpu(acc: Accelerator) -> bool:
    """Check if accelerator is any GPU (NVIDIA or AMD)."""
    return acc in _NVIDIA_VALUES or acc in _AMD_VALUES


def current_accelerator() -> Accelerator:
    """Get the accelerator type for the current compute pool.

    Returns:
        Accelerator type if running in a compute pool, None otherwise.
    """
    from skyward.cluster import instance_info

    pool: InstanceInfo | None = instance_info()
    if pool is None:
        return None
    acc = pool.accelerator
    if acc is None:
        return None
    return acc.get("type")
