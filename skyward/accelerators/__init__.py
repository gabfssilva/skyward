"""Accelerator specifications for Skyward v2.

Provides type-safe accelerator configurations with full IDE autocomplete.
Each accelerator factory returns an immutable Accelerator dataclass
with defaults from the catalog.

Usage:
    import skyward as sky

    # Via namespace (recommended)
    sky.accelerators.H100()              # Default: 80GB, count=1
    sky.accelerators.H100(count=4)       # 4x H100
    sky.accelerators.A100(memory="40GB") # A100 40GB variant

    # Direct import
    from skyward.accelerators import H100, A100, T4
    h100 = H100(count=8)

    # Custom accelerator
    from skyward.accelerators import Custom
    my_gpu = Custom("My-GPU", memory="48GB")

Available accelerators:
    - NVIDIA Datacenter: H100, H200, GH200, B100, B200, GB200,
      A100, A800, A40, A10, A10G, A2, L4, L40, L40S, T4, V100, P100, K80
    - NVIDIA Consumer: RTX_5090-RTX_5060, RTX_4090-RTX_4060,
      RTX_3090-RTX_3050, RTX_2080_Ti-RTX_2060, GTX series
    - NVIDIA Workstation: RTX_A6000-RTX_A2000, Quadro series
    - AMD Instinct: MI300X, MI300A, MI250X, MI250, MI210, MI100, MI50
    - AWS: Trainium1-3, Inferentia1-2
    - Habana: Gaudi, Gaudi2, Gaudi3
    - Google TPU: TPUv2-TPUv6, TPU slices
"""

from __future__ import annotations

# Core dataclass
from .spec import Accelerator

# Literal types for type annotations
from .types import (
    AMD,
    GPU,
    NVIDIA,
    TPU,
    Habana,
    Inferentia,
    Trainium,
)

# =============================================================================
# All factory functions - explicit imports for type safety
# =============================================================================

from .factories import (
    # Custom accelerator
    Custom,
    # NVIDIA Datacenter - Hopper
    H100,
    H200,
    GH200,
    # NVIDIA Datacenter - Blackwell
    B100,
    B200,
    GB200,
    # NVIDIA Datacenter - Ampere
    A100,
    A800,
    A40,
    A10,
    A10G,
    A2,
    # NVIDIA Datacenter - Ada Lovelace
    L4,
    L40,
    L40S,
    # NVIDIA Datacenter - Legacy
    T4,
    V100,
    P100,
    P40,
    P4,
    K80,
    # NVIDIA Consumer RTX - Blackwell (50 series)
    RTX_5090,
    RTX_5080,
    RTX_5070_Ti,
    RTX_5070,
    RTX_5060_Ti,
    RTX_5060,
    # NVIDIA Consumer RTX - Ada Lovelace (40 series)
    RTX_4090,
    RTX_4090D,
    RTX_4080_Super,
    RTX_4080,
    RTX_4070_Ti_Super,
    RTX_4070_Ti,
    RTX_4070_Super,
    RTX_4070,
    RTX_4060_Ti,
    RTX_4060,
    # NVIDIA Consumer RTX - Ampere (30 series)
    RTX_3090_Ti,
    RTX_3090,
    RTX_3080_Ti,
    RTX_3080,
    RTX_3070_Ti,
    RTX_3070,
    RTX_3060_Ti,
    RTX_3060,
    RTX_3060_Laptop,
    RTX_3050,
    # NVIDIA Consumer RTX - Turing (20 series)
    RTX_2080_Ti,
    RTX_2080_Super,
    RTX_2080,
    RTX_2070_Super,
    RTX_2070,
    RTX_2060_Super,
    RTX_2060,
    # NVIDIA Consumer GTX - Turing (16 series)
    GTX_1660_Ti,
    GTX_1660_Super,
    GTX_1660,
    # NVIDIA Consumer GTX - Pascal (10 series)
    GTX_1080_Ti,
    GTX_1080,
    GTX_1070_Ti,
    GTX_1070,
    GTX_1060,
    Titan_Xp,
    Titan_V,
    # NVIDIA Workstation - Ada Lovelace
    RTX_6000_Ada,
    RTX_5880_Ada,
    RTX_5000_Ada,
    # NVIDIA Workstation - Blackwell
    RTX_PRO_6000,
    RTX_PRO_4000,
    # NVIDIA Workstation - Ampere
    RTX_A6000,
    RTX_A5000,
    RTX_A4000,
    RTX_A2000,
    # NVIDIA Workstation - Turing
    Quadro_RTX_8000,
    Quadro_RTX_6000,
    Quadro_RTX_4000,
    Quadro_P4000,
    # AMD Instinct
    MI300X,
    MI300B,
    MI300A,
    MI250X,
    MI250,
    MI210,
    MI100,
    MI50,
    # AMD RadeonPro
    RadeonPro_V710,
    RadeonPro_V520,
    Instinct_MI25,
    # AWS
    Trainium1,
    Trainium2,
    Trainium3,
    Inferentia1,
    Inferentia2,
    # Habana
    Gaudi,
    Gaudi2,
    Gaudi3,
    # Google TPU
    TPUv2,
    TPUv3,
    TPUv4,
    TPUv5e,
    TPUv5p,
    TPUv6,
    # Google TPU Slices
    TPUv2_8,
    TPUv3_8,
    TPUv3_32,
    TPUv4_64,
    TPUv5e_4,
    TPUv5p_8,
)

__all__ = [
    # Core dataclass
    "Accelerator",
    # Literal types for type hints
    "NVIDIA",
    "AMD",
    "TPU",
    "Habana",
    "Trainium",
    "Inferentia",
    "GPU",
    # Custom accelerator
    "Custom",
    # NVIDIA Datacenter - Hopper
    "H100",
    "H200",
    "GH200",
    # NVIDIA Datacenter - Blackwell
    "B100",
    "B200",
    "GB200",
    # NVIDIA Datacenter - Ampere
    "A100",
    "A800",
    "A40",
    "A10",
    "A10G",
    "A2",
    # NVIDIA Datacenter - Ada Lovelace
    "L4",
    "L40",
    "L40S",
    # NVIDIA Datacenter - Legacy
    "T4",
    "V100",
    "P100",
    "P40",
    "P4",
    "K80",
    # NVIDIA Consumer RTX - Blackwell (50 series)
    "RTX_5090",
    "RTX_5080",
    "RTX_5070_Ti",
    "RTX_5070",
    "RTX_5060_Ti",
    "RTX_5060",
    # NVIDIA Consumer RTX - Ada Lovelace (40 series)
    "RTX_4090",
    "RTX_4090D",
    "RTX_4080_Super",
    "RTX_4080",
    "RTX_4070_Ti_Super",
    "RTX_4070_Ti",
    "RTX_4070_Super",
    "RTX_4070",
    "RTX_4060_Ti",
    "RTX_4060",
    # NVIDIA Consumer RTX - Ampere (30 series)
    "RTX_3090_Ti",
    "RTX_3090",
    "RTX_3080_Ti",
    "RTX_3080",
    "RTX_3070_Ti",
    "RTX_3070",
    "RTX_3060_Ti",
    "RTX_3060",
    "RTX_3060_Laptop",
    "RTX_3050",
    # NVIDIA Consumer RTX - Turing (20 series)
    "RTX_2080_Ti",
    "RTX_2080_Super",
    "RTX_2080",
    "RTX_2070_Super",
    "RTX_2070",
    "RTX_2060_Super",
    "RTX_2060",
    # NVIDIA Consumer GTX - Turing (16 series)
    "GTX_1660_Ti",
    "GTX_1660_Super",
    "GTX_1660",
    # NVIDIA Consumer GTX - Pascal (10 series)
    "GTX_1080_Ti",
    "GTX_1080",
    "GTX_1070_Ti",
    "GTX_1070",
    "GTX_1060",
    "Titan_Xp",
    "Titan_V",
    # NVIDIA Workstation - Ada Lovelace
    "RTX_6000_Ada",
    "RTX_5880_Ada",
    "RTX_5000_Ada",
    # NVIDIA Workstation - Blackwell
    "RTX_PRO_6000",
    "RTX_PRO_4000",
    # NVIDIA Workstation - Ampere
    "RTX_A6000",
    "RTX_A5000",
    "RTX_A4000",
    "RTX_A2000",
    # NVIDIA Workstation - Turing
    "Quadro_RTX_8000",
    "Quadro_RTX_6000",
    "Quadro_RTX_4000",
    "Quadro_P4000",
    # AMD Instinct
    "MI300X",
    "MI300B",
    "MI300A",
    "MI250X",
    "MI250",
    "MI210",
    "MI100",
    "MI50",
    # AMD RadeonPro
    "RadeonPro_V710",
    "RadeonPro_V520",
    "Instinct_MI25",
    # AWS
    "Trainium1",
    "Trainium2",
    "Trainium3",
    "Inferentia1",
    "Inferentia2",
    # Habana
    "Gaudi",
    "Gaudi2",
    "Gaudi3",
    # Google TPU
    "TPUv2",
    "TPUv3",
    "TPUv4",
    "TPUv5e",
    "TPUv5p",
    "TPUv6",
    # Google TPU Slices
    "TPUv2_8",
    "TPUv3_8",
    "TPUv3_32",
    "TPUv4_64",
    "TPUv5e_4",
    "TPUv5p_8",
]
