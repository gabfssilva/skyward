"""Hardware specifications catalog for accelerators.

Contains memory, CUDA version ranges, and other metadata for each accelerator.
"""

from __future__ import annotations

import re
from typing import Any, Final


def get_gpu_vram_gb(gpu_model: str) -> int:
    """Get GPU VRAM in GB for a given model.

    Parses the memory field from SPECS and converts to integer GB.

    Args:
        gpu_model: GPU model name (e.g., "A100", "H100", "T4").

    Returns:
        VRAM in GB, or 0 if not found.

    Examples:
        >>> get_gpu_vram_gb("H100")
        80
        >>> get_gpu_vram_gb("A100")
        80
        >>> get_gpu_vram_gb("T4")
        16
    """
    spec = SPECS.get(gpu_model)
    if not spec:
        return 0

    memory = spec.get("memory", "")
    if not memory:
        return 0

    # Parse memory string like "80GB", "16GB", "2TB"
    match = re.match(r"(\d+)(GB|TB)", memory, re.IGNORECASE)
    if not match:
        return 0

    value = int(match.group(1))
    unit = match.group(2).upper()

    if unit == "TB":
        return value * 1024
    return value

SPECS: Final[dict[str, dict[str, Any]]] = {
    # =========================================================================
    # NVIDIA Datacenter - Legacy (Pascal/Volta)
    # =========================================================================
    "K80": {"memory": "12GB", "cuda": {"min": "5.5", "max": "10.2"}},
    "P4": {"memory": "8GB", "cuda": {"min": "8.0", "max": "12.6"}},
    "P40": {"memory": "24GB", "cuda": {"min": "8.0", "max": "12.6"}},
    "P100": {"memory": "16GB", "cuda": {"min": "8.0", "max": "12.6"}},
    "V100": {"memory": "32GB", "cuda": {"min": "9.0", "max": "12.6"}},
    # =========================================================================
    # NVIDIA Datacenter - Turing/Ampere
    # =========================================================================
    "T4": {"memory": "16GB", "cuda": {"min": "10.0", "max": "13.1"}},
    "T4G": {"memory": "16GB", "cuda": {"min": "10.0", "max": "13.1"}},  # ARM64 variant
    "A2": {"memory": "16GB", "cuda": {"min": "11.0", "max": "13.1"}},
    "A10": {"memory": "24GB", "cuda": {"min": "11.0", "max": "13.1"}},
    "A10G": {"memory": "24GB", "cuda": {"min": "11.0", "max": "13.1"}},
    "A40": {"memory": "48GB", "cuda": {"min": "11.1", "max": "13.1"}},
    "A100": {"memory": "80GB", "cuda": {"min": "11.0", "max": "13.1"}},
    "A800": {"memory": "80GB", "cuda": {"min": "11.0", "max": "13.1"}},
    # =========================================================================
    # NVIDIA Datacenter - Ada Lovelace
    # =========================================================================
    "L4": {"memory": "24GB", "cuda": {"min": "11.8", "max": "13.1"}},
    "L40": {"memory": "48GB", "cuda": {"min": "11.8", "max": "13.1"}},
    "L40S": {"memory": "48GB", "cuda": {"min": "11.8", "max": "13.1"}},
    # =========================================================================
    # NVIDIA Datacenter - Hopper
    # =========================================================================
    "H100": {"memory": "80GB", "cuda": {"min": "11.8", "max": "13.1"}},
    "H200": {"memory": "141GB", "cuda": {"min": "11.8", "max": "13.1"}},
    "H200-NVL": {"memory": "141GB", "cuda": {"min": "11.8", "max": "13.1"}},
    "GH200": {"memory": "96GB", "cuda": {"min": "11.8", "max": "13.1"}},
    # =========================================================================
    # NVIDIA Datacenter - Blackwell
    # =========================================================================
    "B100": {"memory": "192GB", "cuda": {"min": "12.8", "max": "13.1"}},
    "B200": {"memory": "192GB", "cuda": {"min": "12.8", "max": "13.1"}},
    "GB200": {"memory": "384GB", "cuda": {"min": "12.8", "max": "13.1"}},
    # =========================================================================
    # NVIDIA Consumer GTX - Pascal (CC 6.1) - DEPRECATED in CUDA 13.0
    # =========================================================================
    "GTX 1060": {"memory": "6GB", "cuda": {"min": "8.0", "max": "12.6"}},
    "GTX 1070": {"memory": "8GB", "cuda": {"min": "8.0", "max": "12.6"}},
    "GTX 1070 Ti": {"memory": "8GB", "cuda": {"min": "8.0", "max": "12.6"}},
    "GTX 1080": {"memory": "8GB", "cuda": {"min": "8.0", "max": "12.6"}},
    "GTX 1080 Ti": {"memory": "11GB", "cuda": {"min": "8.0", "max": "12.6"}},
    "Titan Xp": {"memory": "12GB", "cuda": {"min": "8.0", "max": "12.6"}},
    # =========================================================================
    # NVIDIA Consumer GTX - Turing (CC 7.5)
    # =========================================================================
    "GTX 1660": {"memory": "6GB", "cuda": {"min": "10.0", "max": "13.1"}},
    "GTX 1660 Super": {"memory": "6GB", "cuda": {"min": "10.0", "max": "13.1"}},
    "GTX 1660 Ti": {"memory": "6GB", "cuda": {"min": "10.0", "max": "13.1"}},
    # =========================================================================
    # NVIDIA Consumer RTX - Turing (20 series, CC 7.5)
    # =========================================================================
    "RTX 2060": {"memory": "6GB", "cuda": {"min": "10.0", "max": "13.1"}},
    "RTX 2060 Super": {"memory": "8GB", "cuda": {"min": "10.0", "max": "13.1"}},
    "RTX 2070": {"memory": "8GB", "cuda": {"min": "10.0", "max": "13.1"}},
    "RTX 2070 Super": {"memory": "8GB", "cuda": {"min": "10.0", "max": "13.1"}},
    "RTX 2080": {"memory": "8GB", "cuda": {"min": "10.0", "max": "13.1"}},
    "RTX 2080 Super": {"memory": "8GB", "cuda": {"min": "10.0", "max": "13.1"}},
    "RTX 2080 Ti": {"memory": "11GB", "cuda": {"min": "10.0", "max": "13.1"}},
    # =========================================================================
    # NVIDIA Consumer RTX - Ampere (30 series, CC 8.6)
    # =========================================================================
    "RTX 3050": {"memory": "8GB", "cuda": {"min": "11.1", "max": "13.1"}},
    "RTX 3060": {"memory": "12GB", "cuda": {"min": "11.1", "max": "13.1"}},
    "RTX 3060 Ti": {"memory": "8GB", "cuda": {"min": "11.1", "max": "13.1"}},
    "RTX 3060 Laptop": {"memory": "6GB", "cuda": {"min": "11.1", "max": "13.1"}},
    "RTX 3070": {"memory": "8GB", "cuda": {"min": "11.1", "max": "13.1"}},
    "RTX 3070 Ti": {"memory": "8GB", "cuda": {"min": "11.1", "max": "13.1"}},
    "RTX 3080": {"memory": "10GB", "cuda": {"min": "11.1", "max": "13.1"}},
    "RTX 3080 Ti": {"memory": "12GB", "cuda": {"min": "11.1", "max": "13.1"}},
    "RTX 3090": {"memory": "24GB", "cuda": {"min": "11.1", "max": "13.1"}},
    "RTX 3090 Ti": {"memory": "24GB", "cuda": {"min": "11.1", "max": "13.1"}},
    # =========================================================================
    # NVIDIA Consumer RTX - Ada Lovelace (40 series, CC 8.9)
    # =========================================================================
    "RTX 4060": {"memory": "8GB", "cuda": {"min": "11.8", "max": "13.1"}},
    "RTX 4060 Ti": {"memory": "8GB", "cuda": {"min": "11.8", "max": "13.1"}},
    "RTX 4070": {"memory": "12GB", "cuda": {"min": "11.8", "max": "13.1"}},
    "RTX 4070 Super": {"memory": "12GB", "cuda": {"min": "11.8", "max": "13.1"}},
    "RTX 4070 Ti": {"memory": "12GB", "cuda": {"min": "11.8", "max": "13.1"}},
    "RTX 4070 Ti Super": {"memory": "16GB", "cuda": {"min": "11.8", "max": "13.1"}},
    "RTX 4080": {"memory": "16GB", "cuda": {"min": "11.8", "max": "13.1"}},
    "RTX 4080 Super": {"memory": "16GB", "cuda": {"min": "11.8", "max": "13.1"}},
    "RTX 4090": {"memory": "24GB", "cuda": {"min": "11.8", "max": "13.1"}},
    "RTX 4090D": {"memory": "24GB", "cuda": {"min": "11.8", "max": "13.1"}},
    # =========================================================================
    # NVIDIA Consumer RTX - Blackwell (50 series, CC 10.0)
    # =========================================================================
    "RTX 5060": {"memory": "8GB", "cuda": {"min": "12.8", "max": "13.1"}},
    "RTX 5060 Ti": {"memory": "16GB", "cuda": {"min": "12.8", "max": "13.1"}},
    "RTX 5070": {"memory": "12GB", "cuda": {"min": "12.8", "max": "13.1"}},
    "RTX 5070 Ti": {"memory": "16GB", "cuda": {"min": "12.8", "max": "13.1"}},
    "RTX 5080": {"memory": "16GB", "cuda": {"min": "12.8", "max": "13.1"}},
    "RTX 5090": {"memory": "32GB", "cuda": {"min": "12.8", "max": "13.1"}},
    # =========================================================================
    # NVIDIA Workstation - Pascal (CC 6.1) - DEPRECATED in CUDA 13.0
    # =========================================================================
    "Quadro P4000": {"memory": "8GB", "cuda": {"min": "8.0", "max": "12.6"}},
    # =========================================================================
    # NVIDIA Workstation - Volta (CC 7.0) - DEPRECATED in CUDA 13.0
    # =========================================================================
    "Titan V": {"memory": "12GB", "cuda": {"min": "9.0", "max": "12.6"}},
    # =========================================================================
    # NVIDIA Workstation - Turing (CC 7.5)
    # =========================================================================
    "Quadro RTX 4000": {"memory": "8GB", "cuda": {"min": "10.0", "max": "13.1"}},
    "Quadro RTX 6000": {"memory": "24GB", "cuda": {"min": "10.0", "max": "13.1"}},
    "Quadro RTX 8000": {"memory": "48GB", "cuda": {"min": "10.0", "max": "13.1"}},
    # =========================================================================
    # NVIDIA Workstation - Ampere (CC 8.6)
    # =========================================================================
    "RTX A2000": {"memory": "12GB", "cuda": {"min": "11.1", "max": "13.1"}},
    "RTX A4000": {"memory": "16GB", "cuda": {"min": "11.1", "max": "13.1"}},
    "RTX A5000": {"memory": "24GB", "cuda": {"min": "11.1", "max": "13.1"}},
    "RTX A6000": {"memory": "48GB", "cuda": {"min": "11.1", "max": "13.1"}},
    # =========================================================================
    # NVIDIA Workstation - Ada Lovelace (CC 8.9)
    # =========================================================================
    "RTX 5000 Ada": {"memory": "32GB", "cuda": {"min": "11.8", "max": "13.1"}},
    "RTX 5880 Ada": {"memory": "48GB", "cuda": {"min": "11.8", "max": "13.1"}},
    "RTX 6000 Ada": {"memory": "48GB", "cuda": {"min": "11.8", "max": "13.1"}},
    # =========================================================================
    # NVIDIA Workstation - Blackwell (CC 10.0)
    # =========================================================================
    "RTX PRO 4000": {"memory": "20GB", "cuda": {"min": "12.8", "max": "13.1"}},
    "RTX PRO 6000": {"memory": "96GB", "cuda": {"min": "12.8", "max": "13.1"}},
    # =========================================================================
    # AMD
    # =========================================================================
    "MI50": {"memory": "16GB"},
    "MI100": {"memory": "32GB"},
    "MI210": {"memory": "64GB"},
    "MI250": {"memory": "128GB"},
    "MI250X": {"memory": "128GB"},
    "MI300A": {"memory": "128GB"},
    "MI300B": {"memory": "192GB"},
    "MI300X": {"memory": "192GB"},
    "RadeonPro-V520": {"memory": "8GB"},
    "RadeonPro-V710": {"memory": "16GB"},
    "Instinct-MI25": {"memory": "16GB"},
    # =========================================================================
    # Habana
    # =========================================================================
    "Gaudi": {"memory": "32GB"},
    "Gaudi2": {"memory": "96GB"},
    "Gaudi3": {"memory": "128GB"},
    # =========================================================================
    # AWS
    # =========================================================================
    "Trainium1": {"memory": "32GB"},
    "Trainium2": {"memory": "64GB"},
    "Trainium3": {"memory": "128GB"},
    "Inferentia1": {"memory": "8GB"},
    "Inferentia2": {"memory": "32GB"},
    # =========================================================================
    # Google TPU
    # =========================================================================
    "TPUv2": {"memory": "8GB"},
    "TPUv3": {"memory": "16GB"},
    "TPUv4": {"memory": "32GB"},
    "TPUv5e": {"memory": "16GB"},
    "TPUv5p": {"memory": "95GB"},
    "TPUv6": {"memory": "32GB"},
    "TPUv2-8": {"memory": "64GB", "count": 8},
    "TPUv3-8": {"memory": "128GB", "count": 8},
    "TPUv3-32": {"memory": "512GB", "count": 32},
    "TPUv4-64": {"memory": "2TB", "count": 64},
    "TPUv5e-4": {"memory": "64GB", "count": 4},
    "TPUv5p-8": {"memory": "760GB", "count": 8},
}
