"""Hardware specifications catalog for accelerators.

Contains memory, CUDA version ranges, manufacturer, architecture, and other
metadata for each accelerator.  This is the single source of truth used by
both the runtime API and the documentation catalog pipeline.
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
    # NVIDIA Datacenter - Kepler
    # =========================================================================
    "K80": {"memory": "12GB", "cuda": {"min": "5.5", "max": "10.2"}, "manufacturer": "NVIDIA", "architecture": "Kepler"},
    # =========================================================================
    # NVIDIA Datacenter - Pascal
    # =========================================================================
    "P4": {"memory": "8GB", "cuda": {"min": "8.0", "max": "12.6"}, "manufacturer": "NVIDIA", "architecture": "Pascal"},
    "P40": {"memory": "24GB", "cuda": {"min": "8.0", "max": "12.6"}, "manufacturer": "NVIDIA", "architecture": "Pascal"},
    "P100": {"memory": "16GB", "cuda": {"min": "8.0", "max": "12.6"}, "manufacturer": "NVIDIA", "architecture": "Pascal"},
    # =========================================================================
    # NVIDIA Datacenter - Volta
    # =========================================================================
    "V100": {"memory": "32GB", "cuda": {"min": "9.0", "max": "12.6"}, "manufacturer": "NVIDIA", "architecture": "Volta"},
    # =========================================================================
    # NVIDIA Datacenter - Turing
    # =========================================================================
    "T4": {"memory": "16GB", "cuda": {"min": "10.0", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Turing"},
    "T4G": {"memory": "16GB", "cuda": {"min": "10.0", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Turing"},  # ARM64 variant
    # =========================================================================
    # NVIDIA Datacenter - Ampere
    # =========================================================================
    "A2": {"memory": "16GB", "cuda": {"min": "11.0", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ampere"},
    "A10": {"memory": "24GB", "cuda": {"min": "11.0", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ampere"},
    "A10G": {"memory": "24GB", "cuda": {"min": "11.0", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ampere"},
    "A16": {"memory": "16GB", "cuda": {"min": "11.1", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ampere"},
    "A30": {"memory": "24GB", "cuda": {"min": "11.1", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ampere"},
    "A40": {"memory": "48GB", "cuda": {"min": "11.1", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ampere"},
    "A100": {"memory": "80GB", "cuda": {"min": "11.0", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ampere"},
    "A100X": {"memory": "80GB", "cuda": {"min": "11.0", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ampere"},
    "A800": {"memory": "80GB", "cuda": {"min": "11.0", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ampere"},
    # =========================================================================
    # NVIDIA Datacenter - Ada Lovelace
    # =========================================================================
    "L4": {"memory": "24GB", "cuda": {"min": "11.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ada Lovelace"},
    "L40": {"memory": "48GB", "cuda": {"min": "11.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ada Lovelace"},
    "L40S": {"memory": "48GB", "cuda": {"min": "11.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ada Lovelace"},
    # =========================================================================
    # NVIDIA Datacenter - Hopper
    # =========================================================================
    "H100": {"memory": "80GB", "cuda": {"min": "11.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Hopper"},
    "H100-NVL": {"memory": "94GB", "cuda": {"min": "11.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Hopper"},
    "H200": {"memory": "141GB", "cuda": {"min": "11.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Hopper"},
    "H200-NVL": {"memory": "141GB", "cuda": {"min": "11.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Hopper"},
    "GH200": {"memory": "96GB", "cuda": {"min": "11.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Hopper"},
    # =========================================================================
    # NVIDIA Datacenter - Blackwell
    # =========================================================================
    "B100": {"memory": "192GB", "cuda": {"min": "12.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Blackwell"},
    "B200": {"memory": "192GB", "cuda": {"min": "12.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Blackwell"},
    "B300": {"memory": "288GB", "cuda": {"min": "12.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Blackwell"},
    "GB200": {"memory": "384GB", "cuda": {"min": "12.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Blackwell"},
    "GB300": {"memory": "288GB", "cuda": {"min": "12.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Blackwell"},
    # =========================================================================
    # NVIDIA Consumer GTX - Pascal (CC 6.1) - DEPRECATED in CUDA 13.0
    # =========================================================================
    "GTX 1050": {"memory": "2GB", "cuda": {"min": "8.0", "max": "12.6"}, "manufacturer": "NVIDIA", "architecture": "Pascal"},
    "GTX 1050 Ti": {"memory": "4GB", "cuda": {"min": "8.0", "max": "12.6"}, "manufacturer": "NVIDIA", "architecture": "Pascal"},
    "GTX 1060": {"memory": "6GB", "cuda": {"min": "8.0", "max": "12.6"}, "manufacturer": "NVIDIA", "architecture": "Pascal"},
    "GTX 1070": {"memory": "8GB", "cuda": {"min": "8.0", "max": "12.6"}, "manufacturer": "NVIDIA", "architecture": "Pascal"},
    "GTX 1070 Ti": {"memory": "8GB", "cuda": {"min": "8.0", "max": "12.6"}, "manufacturer": "NVIDIA", "architecture": "Pascal"},
    "GTX 1080": {"memory": "8GB", "cuda": {"min": "8.0", "max": "12.6"}, "manufacturer": "NVIDIA", "architecture": "Pascal"},
    "GTX 1080 Ti": {"memory": "11GB", "cuda": {"min": "8.0", "max": "12.6"}, "manufacturer": "NVIDIA", "architecture": "Pascal"},
    "Titan Xp": {"memory": "12GB", "cuda": {"min": "8.0", "max": "12.6"}, "manufacturer": "NVIDIA", "architecture": "Pascal"},
    # =========================================================================
    # NVIDIA Consumer GTX - Turing (CC 7.5)
    # =========================================================================
    "GTX 1650": {"memory": "4GB", "cuda": {"min": "10.0", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Turing"},
    "GTX 1660": {"memory": "6GB", "cuda": {"min": "10.0", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Turing"},
    "GTX 1660 Super": {"memory": "6GB", "cuda": {"min": "10.0", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Turing"},
    "GTX 1660 Ti": {"memory": "6GB", "cuda": {"min": "10.0", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Turing"},
    # =========================================================================
    # NVIDIA Consumer RTX - Turing (20 series, CC 7.5)
    # =========================================================================
    "RTX 2060": {"memory": "6GB", "cuda": {"min": "10.0", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Turing"},
    "RTX 2060 Super": {"memory": "8GB", "cuda": {"min": "10.0", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Turing"},
    "RTX 2070": {"memory": "8GB", "cuda": {"min": "10.0", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Turing"},
    "RTX 2070 Super": {"memory": "8GB", "cuda": {"min": "10.0", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Turing"},
    "RTX 2080": {"memory": "8GB", "cuda": {"min": "10.0", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Turing"},
    "RTX 2080 Super": {"memory": "8GB", "cuda": {"min": "10.0", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Turing"},
    "RTX 2080 Ti": {"memory": "11GB", "cuda": {"min": "10.0", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Turing"},
    "Titan RTX": {"memory": "24GB", "cuda": {"min": "10.0", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Turing"},
    # =========================================================================
    # NVIDIA Consumer RTX - Ampere (30 series, CC 8.6)
    # =========================================================================
    "RTX 3050": {"memory": "8GB", "cuda": {"min": "11.1", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ampere"},
    "RTX 3060": {"memory": "12GB", "cuda": {"min": "11.1", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ampere"},
    "RTX 3060 Ti": {"memory": "8GB", "cuda": {"min": "11.1", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ampere"},
    "RTX 3060 Laptop": {"memory": "6GB", "cuda": {"min": "11.1", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ampere"},
    "RTX 3070": {"memory": "8GB", "cuda": {"min": "11.1", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ampere"},
    "RTX 3070 Ti": {"memory": "8GB", "cuda": {"min": "11.1", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ampere"},
    "RTX 3080": {"memory": "10GB", "cuda": {"min": "11.1", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ampere"},
    "RTX 3080 Ti": {"memory": "12GB", "cuda": {"min": "11.1", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ampere"},
    "RTX 3090": {"memory": "24GB", "cuda": {"min": "11.1", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ampere"},
    "RTX 3090 Ti": {"memory": "24GB", "cuda": {"min": "11.1", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ampere"},
    # =========================================================================
    # NVIDIA Consumer RTX - Ada Lovelace (40 series, CC 8.9)
    # =========================================================================
    "RTX 4060": {"memory": "8GB", "cuda": {"min": "11.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ada Lovelace"},
    "RTX 4060 Ti": {"memory": "8GB", "cuda": {"min": "11.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ada Lovelace"},
    "RTX 4070": {"memory": "12GB", "cuda": {"min": "11.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ada Lovelace"},
    "RTX 4070 Super": {"memory": "12GB", "cuda": {"min": "11.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ada Lovelace"},
    "RTX 4070 Ti": {"memory": "12GB", "cuda": {"min": "11.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ada Lovelace"},
    "RTX 4070 Ti Super": {"memory": "16GB", "cuda": {"min": "11.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ada Lovelace"},
    "RTX 4080": {"memory": "16GB", "cuda": {"min": "11.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ada Lovelace"},
    "RTX 4080 Super": {"memory": "16GB", "cuda": {"min": "11.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ada Lovelace"},
    "RTX 4090": {"memory": "24GB", "cuda": {"min": "11.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ada Lovelace"},
    "RTX 4090D": {"memory": "24GB", "cuda": {"min": "11.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ada Lovelace"},
    # =========================================================================
    # NVIDIA Consumer RTX - Blackwell (50 series, CC 10.0)
    # =========================================================================
    "RTX 5060": {"memory": "8GB", "cuda": {"min": "12.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Blackwell"},
    "RTX 5060 Ti": {"memory": "16GB", "cuda": {"min": "12.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Blackwell"},
    "RTX 5070": {"memory": "12GB", "cuda": {"min": "12.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Blackwell"},
    "RTX 5070 Ti": {"memory": "16GB", "cuda": {"min": "12.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Blackwell"},
    "RTX 5080": {"memory": "16GB", "cuda": {"min": "12.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Blackwell"},
    "RTX 5090": {"memory": "32GB", "cuda": {"min": "12.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Blackwell"},
    # =========================================================================
    # NVIDIA Mining (CMP) - Turing
    # =========================================================================
    "CMP 50HX": {"memory": "10GB", "cuda": {"min": "11.0", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Turing"},
    # =========================================================================
    # NVIDIA Workstation - Pascal (CC 6.1) - DEPRECATED in CUDA 13.0
    # =========================================================================
    "Quadro P2000": {"memory": "5GB", "cuda": {"min": "8.0", "max": "12.6"}, "manufacturer": "NVIDIA", "architecture": "Pascal"},
    "Quadro P4000": {"memory": "8GB", "cuda": {"min": "8.0", "max": "12.6"}, "manufacturer": "NVIDIA", "architecture": "Pascal"},
    # =========================================================================
    # NVIDIA Workstation - Volta (CC 7.0) - DEPRECATED in CUDA 13.0
    # =========================================================================
    "Titan V": {"memory": "12GB", "cuda": {"min": "9.0", "max": "12.6"}, "manufacturer": "NVIDIA", "architecture": "Volta"},
    # =========================================================================
    # NVIDIA Workstation - Turing (CC 7.5)
    # =========================================================================
    "Quadro RTX 4000": {"memory": "8GB", "cuda": {"min": "10.0", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Turing"},
    "Quadro RTX 5000": {"memory": "16GB", "cuda": {"min": "10.0", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Turing"},
    "Quadro RTX 6000": {"memory": "24GB", "cuda": {"min": "10.0", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Turing"},
    "Quadro RTX 8000": {"memory": "48GB", "cuda": {"min": "10.0", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Turing"},
    # =========================================================================
    # NVIDIA Workstation - Ampere (CC 8.6)
    # =========================================================================
    "RTX A2000": {"memory": "12GB", "cuda": {"min": "11.1", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ampere"},
    "RTX A4000": {"memory": "16GB", "cuda": {"min": "11.1", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ampere"},
    "RTX A4500": {"memory": "20GB", "cuda": {"min": "11.1", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ampere"},
    "RTX A5000": {"memory": "24GB", "cuda": {"min": "11.1", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ampere"},
    "RTX A5000 Pro": {"memory": "24GB", "cuda": {"min": "11.1", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ampere"},
    "RTX A6000": {"memory": "48GB", "cuda": {"min": "11.1", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ampere"},
    # =========================================================================
    # NVIDIA Workstation - Ada Lovelace (CC 8.9)
    # =========================================================================
    "RTX 2000 Ada": {"memory": "16GB", "cuda": {"min": "11.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ada Lovelace"},
    "RTX 4000 Ada": {"memory": "20GB", "cuda": {"min": "11.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ada Lovelace"},
    "RTX 4500 Ada": {"memory": "24GB", "cuda": {"min": "11.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ada Lovelace"},
    "RTX 5000 Ada": {"memory": "32GB", "cuda": {"min": "11.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ada Lovelace"},
    "RTX 5880 Ada": {"memory": "48GB", "cuda": {"min": "11.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ada Lovelace"},
    "RTX 6000 Ada": {"memory": "48GB", "cuda": {"min": "11.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Ada Lovelace"},
    # =========================================================================
    # NVIDIA Workstation - Blackwell (CC 10.0)
    # =========================================================================
    "RTX PRO 4000": {"memory": "24GB", "cuda": {"min": "12.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Blackwell"},
    "RTX PRO 4500": {"memory": "32GB", "cuda": {"min": "12.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Blackwell"},
    "RTX PRO 5000": {"memory": "48GB", "cuda": {"min": "12.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Blackwell"},
    "RTX PRO 6000": {"memory": "96GB", "cuda": {"min": "12.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Blackwell"},
    "RTX PRO 6000 MaxQ": {"memory": "96GB", "cuda": {"min": "12.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Blackwell"},
    "RTX PRO 6000 S": {"memory": "96GB", "cuda": {"min": "12.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Blackwell"},
    "RTX PRO 6000 WK": {"memory": "96GB", "cuda": {"min": "12.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Blackwell"},
    "RTX PRO 6000 WS": {"memory": "96GB", "cuda": {"min": "12.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Blackwell"},
    "RTX PRO Server 6000": {"memory": "96GB", "cuda": {"min": "12.8", "max": "13.1"}, "manufacturer": "NVIDIA", "architecture": "Blackwell"},
    # =========================================================================
    # AMD - CDNA
    # =========================================================================
    "MI50": {"memory": "16GB", "manufacturer": "AMD", "architecture": "CDNA"},
    "MI100": {"memory": "32GB", "manufacturer": "AMD", "architecture": "CDNA"},
    # =========================================================================
    # AMD - CDNA2
    # =========================================================================
    "MI210": {"memory": "64GB", "manufacturer": "AMD", "architecture": "CDNA2"},
    "MI250": {"memory": "128GB", "manufacturer": "AMD", "architecture": "CDNA2"},
    "MI250X": {"memory": "128GB", "manufacturer": "AMD", "architecture": "CDNA2"},
    # =========================================================================
    # AMD - CDNA3
    # =========================================================================
    "MI300A": {"memory": "128GB", "manufacturer": "AMD", "architecture": "CDNA3"},
    "MI300B": {"memory": "192GB", "manufacturer": "AMD", "architecture": "CDNA3"},
    "MI300X": {"memory": "192GB", "manufacturer": "AMD", "architecture": "CDNA3"},
    # =========================================================================
    # AMD - RDNA
    # =========================================================================
    "Radeon Pro V520": {"memory": "8GB", "manufacturer": "AMD", "architecture": "RDNA"},
    "RadeonPro-V520": {"memory": "8GB", "manufacturer": "AMD", "architecture": "RDNA"},
    "RadeonPro-V710": {"memory": "16GB", "manufacturer": "AMD", "architecture": "RDNA2"},
    "Instinct-MI25": {"memory": "16GB", "manufacturer": "AMD", "architecture": "GCN"},
    # =========================================================================
    # Intel - Habana
    # =========================================================================
    "Gaudi": {"memory": "32GB", "manufacturer": "Intel", "architecture": "Gaudi"},
    "Gaudi2": {"memory": "96GB", "manufacturer": "Intel", "architecture": "Gaudi"},
    "Gaudi3": {"memory": "128GB", "manufacturer": "Intel", "architecture": "Gaudi"},
    # =========================================================================
    # AWS
    # =========================================================================
    "Trainium1": {"memory": "32GB", "manufacturer": "AWS", "architecture": "Trainium"},
    "Trainium2": {"memory": "64GB", "manufacturer": "AWS", "architecture": "Trainium"},
    "Trainium3": {"memory": "128GB", "manufacturer": "AWS", "architecture": "Trainium"},
    "Inferentia1": {"memory": "8GB", "manufacturer": "AWS", "architecture": "Inferentia"},
    "Inferentia2": {"memory": "32GB", "manufacturer": "AWS", "architecture": "Inferentia"},
    # =========================================================================
    # Google TPU
    # =========================================================================
    "TPUv2": {"memory": "8GB", "manufacturer": "Google", "architecture": "TPU"},
    "TPUv3": {"memory": "16GB", "manufacturer": "Google", "architecture": "TPU"},
    "TPUv4": {"memory": "32GB", "manufacturer": "Google", "architecture": "TPU"},
    "TPUv5e": {"memory": "16GB", "manufacturer": "Google", "architecture": "TPU"},
    "TPUv5p": {"memory": "95GB", "manufacturer": "Google", "architecture": "TPU"},
    "TPUv6": {"memory": "32GB", "manufacturer": "Google", "architecture": "TPU"},
    "TPUv2-8": {"memory": "64GB", "count": 8, "manufacturer": "Google", "architecture": "TPU"},
    "TPUv3-8": {"memory": "128GB", "count": 8, "manufacturer": "Google", "architecture": "TPU"},
    "TPUv3-32": {"memory": "512GB", "count": 32, "manufacturer": "Google", "architecture": "TPU"},
    "TPUv4-64": {"memory": "2TB", "count": 64, "manufacturer": "Google", "architecture": "TPU"},
    "TPUv5e-4": {"memory": "64GB", "count": 4, "manufacturer": "Google", "architecture": "TPU"},
    "TPUv5p-8": {"memory": "760GB", "count": 8, "manufacturer": "Google", "architecture": "TPU"},
}
