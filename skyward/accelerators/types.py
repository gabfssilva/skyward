"""Literal types for accelerator specifications.

Provides type-safe accelerator names for autocomplete in IDEs.
"""

from typing import Literal

# =============================================================================
# NVIDIA GPUs
# =============================================================================

NVIDIA = Literal[
    # Legacy - Pascal (CC 6.1) - CUDA 8.0-12.6
    "K80",
    "P4",
    "P40",
    "P100",
    "Quadro P4000",
    "Titan Xp",
    # Legacy - Volta (CC 7.0) - CUDA 9.0-12.6
    "V100",
    "Titan V",
    # Mid-range - Turing/Ampere
    "T4",
    "A2",
    "A10",
    "A10G",
    # Datacenter - Ampere (CC 8.0/8.6) - CUDA 11.0-13.x
    "A40",
    "A100-40",
    "A100-80",
    "A100-40GB",
    "A100-80GB",
    "A800",
    # Enterprise - Ada Lovelace (CC 8.9) - CUDA 11.8-13.x
    "L4",
    "L40",
    "L40S",
    # Hopper (CC 9.0) - CUDA 11.8-13.x
    "H100-80GB",
    "H100-NVL",
    "H100-SXM",
    "H100-PCIe",
    "H200",
    "H200-NVL",
    # Blackwell (CC 10.0) - CUDA 12.8-13.x
    "B100",
    "B200",
    "GB200",
    # Grace Hopper
    "GH200",
    # Consumer GTX - Pascal (CC 6.1) - CUDA 8.0-12.6
    "GTX 1060",
    "GTX 1070",
    "GTX 1070 Ti",
    "GTX 1080",
    "GTX 1080 Ti",
    # Consumer GTX - Turing (CC 7.5) - CUDA 10.0-13.x
    "GTX 1660",
    "GTX 1660 Super",
    "GTX 1660 Ti",
    # Consumer RTX - Turing (20 series) - CUDA 10.0-13.x
    "RTX 2060",
    "RTX 2060 Super",
    "RTX 2070",
    "RTX 2070 Super",
    "RTX 2080",
    "RTX 2080 Super",
    "RTX 2080 Ti",
    # Consumer RTX - Ampere (30 series) - CUDA 11.1-13.x
    "RTX 3050",
    "RTX 3060",
    "RTX 3060 Ti",
    "RTX 3060 Laptop",
    "RTX 3070",
    "RTX 3070 Ti",
    "RTX 3080",
    "RTX 3080 Ti",
    "RTX 3090",
    "RTX 3090 Ti",
    # Consumer RTX - Ada Lovelace (40 series) - CUDA 11.8-13.x
    "RTX 4060",
    "RTX 4060 Ti",
    "RTX 4070",
    "RTX 4070 Super",
    "RTX 4070 Ti",
    "RTX 4070 Ti Super",
    "RTX 4080",
    "RTX 4080 Super",
    "RTX 4090",
    "RTX 4090D",
    # Consumer RTX - Blackwell (50 series) - CUDA 12.8-13.x
    "RTX 5060",
    "RTX 5060 Ti",
    "RTX 5070",
    "RTX 5070 Ti",
    "RTX 5080",
    "RTX 5090",
    # Workstation - Turing (CC 7.5) - CUDA 10.0-13.x
    "Quadro RTX 4000",
    "Quadro RTX 6000",
    "Quadro RTX 8000",
    # Workstation - Ampere (CC 8.6) - CUDA 11.1-13.x
    "RTX A2000",
    "RTX A4000",
    "RTX A5000",
    "RTX A6000",
    # Workstation - Ada Lovelace (CC 8.9) - CUDA 11.8-13.x
    "RTX 5000 Ada",
    "RTX 5880 Ada",
    "RTX 6000 Ada",
    # Workstation - Blackwell (CC 10.0) - CUDA 12.8-13.x
    "RTX PRO 4000",
    "RTX PRO 6000",
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
