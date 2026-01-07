"""Literal types for accelerator specifications.

Provides type-safe accelerator names for autocomplete in IDEs.
"""

from typing import Literal

# =============================================================================
# NVIDIA GPUs
# =============================================================================

NVIDIA = Literal[
    # Legacy
    "K80",
    "M60",
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
    # Consumer RTX - Turing (20 series)
    "RTX 2060",
    "RTX 2060 Super",
    "RTX 2070",
    "RTX 2070 Super",
    "RTX 2080",
    "RTX 2080 Super",
    "RTX 2080 Ti",
    # Consumer RTX - Ampere (30 series)
    "RTX 3060",
    "RTX 3060 Ti",
    "RTX 3070",
    "RTX 3070 Ti",
    "RTX 3080",
    "RTX 3080 Ti",
    "RTX 3090",
    "RTX 3090 Ti",
    # Consumer RTX - Ada Lovelace (40 series)
    "RTX 4060",
    "RTX 4060 Ti",
    "RTX 4070",
    "RTX 4070 Super",
    "RTX 4070 Ti",
    "RTX 4070 Ti Super",
    "RTX 4080",
    "RTX 4080 Super",
    "RTX 4090",
    # Consumer RTX - Blackwell (50 series)
    "RTX 5070",
    "RTX 5070 Ti",
    "RTX 5080",
    "RTX 5090",
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
    # Instinct (data center compute)
    "MI50",
    "MI100",
    "MI210",
    "MI250",
    "MI250X",
    "MI300A",
    "MI300B",
    "MI300X",
    "Instinct-MI25",
    # Radeon Pro (graphics/streaming)
    "RadeonPro-V520",
    "RadeonPro-V710",
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

GPU = NVIDIA | AMD | None
