"""Hardware specifications catalog for accelerators.

Contains memory, CUDA version ranges, and other metadata for each accelerator.
"""

from typing import Any, Final

SPECS: Final[dict[str, dict[str, Any]]] = {
    # =========================================================================
    # NVIDIA - single memory option
    # =========================================================================
    "T4": {"memory": "16GB", "cuda": {"min": "10.0", "max": "13.1"}},
    "L4": {"memory": "24GB", "cuda": {"min": "11.8", "max": "13.1"}},
    "L40": {"memory": "48GB", "cuda": {"min": "11.8", "max": "13.1"}},
    "L40S": {"memory": "48GB", "cuda": {"min": "11.8", "max": "13.1"}},
    "P100": {"memory": "16GB", "cuda": {"min": "8.0", "max": "12.6"}},
    "P4": {"memory": "8GB", "cuda": {"min": "8.0", "max": "12.6"}},
    "K80": {"memory": "12GB", "cuda": {"min": "5.5", "max": "10.2"}},
    "A10": {"memory": "24GB", "cuda": {"min": "11.0", "max": "13.1"}},
    "A10G": {"memory": "24GB", "cuda": {"min": "11.0", "max": "13.1"}},
    "A2": {"memory": "16GB", "cuda": {"min": "11.0", "max": "13.1"}},
    "H200": {"memory": "141GB", "cuda": {"min": "11.8", "max": "13.1"}},
    "B100": {"memory": "192GB", "cuda": {"min": "12.8", "max": "13.1"}},
    "B200": {"memory": "192GB", "cuda": {"min": "12.8", "max": "13.1"}},
    "GB200": {"memory": "384GB", "cuda": {"min": "12.8", "max": "13.1"}},
    "GH200": {"memory": "96GB", "cuda": {"min": "11.8", "max": "13.1"}},
    # =========================================================================
    # NVIDIA - multiple memory options (defaults handled in factory)
    # =========================================================================
    "H100": {"memory": "80GB", "cuda": {"min": "11.8", "max": "13.1"}},
    "A100": {"memory": "80GB", "cuda": {"min": "11.0", "max": "13.1"}},
    "V100": {"memory": "32GB", "cuda": {"min": "9.0", "max": "12.6"}},
    # =========================================================================
    # NVIDIA RTX - Turing (20 series)
    # =========================================================================
    "RTX 2060": {"memory": "6GB", "cuda": {"min": "10.0", "max": "12.6"}},
    "RTX 2060 Super": {"memory": "8GB", "cuda": {"min": "10.0", "max": "12.6"}},
    "RTX 2070": {"memory": "8GB", "cuda": {"min": "10.0", "max": "12.6"}},
    "RTX 2070 Super": {"memory": "8GB", "cuda": {"min": "10.0", "max": "12.6"}},
    "RTX 2080": {"memory": "8GB", "cuda": {"min": "10.0", "max": "12.6"}},
    "RTX 2080 Super": {"memory": "8GB", "cuda": {"min": "10.0", "max": "12.6"}},
    "RTX 2080 Ti": {"memory": "11GB", "cuda": {"min": "10.0", "max": "12.6"}},
    # =========================================================================
    # NVIDIA RTX - Ampere (30 series)
    # =========================================================================
    "RTX 3060": {"memory": "12GB", "cuda": {"min": "11.1", "max": "13.1"}},
    "RTX 3060 Ti": {"memory": "8GB", "cuda": {"min": "11.1", "max": "13.1"}},
    "RTX 3070": {"memory": "8GB", "cuda": {"min": "11.1", "max": "13.1"}},
    "RTX 3070 Ti": {"memory": "8GB", "cuda": {"min": "11.1", "max": "13.1"}},
    "RTX 3080": {"memory": "10GB", "cuda": {"min": "11.1", "max": "13.1"}},
    "RTX 3080 Ti": {"memory": "12GB", "cuda": {"min": "11.1", "max": "13.1"}},
    "RTX 3090": {"memory": "24GB", "cuda": {"min": "11.1", "max": "13.1"}},
    "RTX 3090 Ti": {"memory": "24GB", "cuda": {"min": "11.1", "max": "13.1"}},
    # =========================================================================
    # NVIDIA RTX - Ada Lovelace (40 series)
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
    # =========================================================================
    # NVIDIA RTX - Blackwell (50 series)
    # =========================================================================
    "RTX 5070": {"memory": "12GB", "cuda": {"min": "12.8", "max": "13.1"}},
    "RTX 5070 Ti": {"memory": "16GB", "cuda": {"min": "12.8", "max": "13.1"}},
    "RTX 5080": {"memory": "16GB", "cuda": {"min": "12.8", "max": "13.1"}},
    "RTX 5090": {"memory": "32GB", "cuda": {"min": "12.8", "max": "13.1"}},
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
