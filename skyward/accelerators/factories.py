"""Accelerator factory functions.

Provides type-safe factory functions for all supported accelerators.
Each factory returns an Accelerator instance with appropriate defaults.

Usage:
    from skyward.accelerators import H100, A100, T4

    h100 = H100()                     # Default: 80GB, count=1
    h100_x4 = H100(count=4)           # 4x H100
    a100_40 = A100(memory="40GB")     # A100 40GB variant
"""

from __future__ import annotations

from typing import Literal

from .catalog import SPECS

from .spec import Accelerator


# =============================================================================
# NVIDIA Datacenter - Hopper
# =============================================================================


def H100(
    *,
    memory: Literal["40GB", "80GB"] = "80GB",
    form_factor: Literal["SXM", "PCIe", "NVL"] | None = None,
    count: int = 1,
) -> Accelerator:
    """NVIDIA H100 - Hopper architecture (2022).

    The H100 is NVIDIA's flagship datacenter GPU for AI/ML workloads.
    Supports FP8, with massive improvements for transformer models.

    Args:
        memory: VRAM size - 40GB (PCIe) or 80GB (SXM/NVL).
        form_factor: SXM (high bandwidth), PCIe, or NVL (2x GPU module).
        count: Number of GPUs per node.

    Returns:
        Accelerator specification for H100.
    """
    name = "H100"
    actual_memory = memory

    if form_factor == "NVL":
        name = "H200-NVL"
        actual_memory = "94GB"

    return Accelerator(
        name=name,
        memory=actual_memory,
        count=count,
        metadata={"cuda": SPECS["H100"]["cuda"], "form_factor": form_factor},
    )


def H200(
    *,
    memory: Literal["141GB"] = "141GB",
    form_factor: Literal["SXM", "NVL"] | None = None,
    count: int = 1,
) -> Accelerator:
    """NVIDIA H200 - Hopper architecture with HBM3e (2024).

    H200 features 141GB HBM3e memory with 4.8TB/s bandwidth.
    Drop-in replacement for H100 with 1.4-1.9x inference speedup.

    Args:
        memory: VRAM size (141GB for standard, varies for NVL).
        form_factor: SXM or NVL variant.
        count: Number of GPUs per node.

    Returns:
        Accelerator specification for H200.
    """
    name = "H200-NVL" if form_factor == "NVL" else "H200"
    return Accelerator(
        name=name,
        memory=memory,
        count=count,
        metadata={"cuda": SPECS["H200"]["cuda"], "form_factor": form_factor},
    )


def GH200(*, count: int = 1) -> Accelerator:
    """NVIDIA Grace Hopper Superchip (2023).

    Combines Grace CPU with Hopper GPU via NVLink-C2C.
    96GB HBM3 for GPU, 480GB LPDDR5X for CPU.

    Args:
        count: Number of superchips per node.

    Returns:
        Accelerator specification for GH200.
    """
    return Accelerator.from_name("GH200", count=count)


# =============================================================================
# NVIDIA Datacenter - Blackwell
# =============================================================================


def B100(*, count: int = 1) -> Accelerator:
    """NVIDIA B100 - Blackwell architecture (2024).

    Second-generation transformer engine with FP4 support.
    192GB HBM3e memory.

    Args:
        count: Number of GPUs per node.

    Returns:
        Accelerator specification for B100.
    """
    return Accelerator.from_name("B100", count=count)


def B200(*, count: int = 1) -> Accelerator:
    """NVIDIA B200 - Blackwell architecture (2024).

    Flagship Blackwell GPU with 192GB HBM3e.
    Up to 2.5x inference performance vs H100.

    Args:
        count: Number of GPUs per node.

    Returns:
        Accelerator specification for B200.
    """
    return Accelerator.from_name("B200", count=count)


def GB200(*, count: int = 1) -> Accelerator:
    """NVIDIA Grace Blackwell Superchip (2024).

    Combines Grace CPU with two Blackwell GPUs.
    384GB combined HBM3e memory.

    Args:
        count: Number of superchips per node.

    Returns:
        Accelerator specification for GB200.
    """
    return Accelerator.from_name("GB200", count=count)


# =============================================================================
# NVIDIA Datacenter - Ampere
# =============================================================================


def A100(
    *,
    memory: Literal["40GB", "80GB"] = "80GB",
    count: int = 1,
) -> Accelerator:
    """NVIDIA A100 - Ampere architecture (2020).

    The A100 was the first GPU with TF32 and structural sparsity.
    Available in 40GB PCIe and 80GB SXM variants.

    Args:
        memory: VRAM size - 40GB or 80GB.
        count: Number of GPUs per node.

    Returns:
        Accelerator specification for A100.
    """
    return Accelerator(
        name="A100",
        memory=memory,
        count=count,
        metadata={"cuda": SPECS["A100"]["cuda"]},
    )


def A800(*, count: int = 1) -> Accelerator:
    """NVIDIA A800 - China-specific A100 variant.

    Reduced NVLink bandwidth to comply with export restrictions.
    Same 80GB HBM2e memory as A100.

    Args:
        count: Number of GPUs per node.

    Returns:
        Accelerator specification for A800.
    """
    return Accelerator.from_name("A800", count=count)


def A40(*, count: int = 1) -> Accelerator:
    """NVIDIA A40 - Professional visualization + compute (2020).

    48GB GDDR6 memory, optimized for virtual workstations.

    Args:
        count: Number of GPUs per node.

    Returns:
        Accelerator specification for A40.
    """
    return Accelerator.from_name("A40", count=count)


def A10(*, count: int = 1) -> Accelerator:
    """NVIDIA A10 - Inference-optimized Ampere GPU.

    24GB GDDR6 memory, popular for inference workloads.

    Args:
        count: Number of GPUs per node.

    Returns:
        Accelerator specification for A10.
    """
    return Accelerator.from_name("A10", count=count)


def A10G(*, count: int = 1) -> Accelerator:
    """NVIDIA A10G - AWS-specific A10 variant.

    24GB GDDR6 memory, available on g5 instances.

    Args:
        count: Number of GPUs per node.

    Returns:
        Accelerator specification for A10G.
    """
    return Accelerator.from_name("A10G", count=count)


def A2(*, count: int = 1) -> Accelerator:
    """NVIDIA A2 - Entry-level Ampere for inference.

    16GB GDDR6 memory, low power consumption.

    Args:
        count: Number of GPUs per node.

    Returns:
        Accelerator specification for A2.
    """
    return Accelerator.from_name("A2", count=count)


# =============================================================================
# NVIDIA Datacenter - Ada Lovelace
# =============================================================================


def L4(*, count: int = 1) -> Accelerator:
    """NVIDIA L4 - Ada Lovelace inference GPU (2023).

    24GB GDDR6 memory, replaces T4 for inference.
    Excellent price/performance for video and LLM inference.

    Args:
        count: Number of GPUs per node.

    Returns:
        Accelerator specification for L4.
    """
    return Accelerator.from_name("L4", count=count)


def L40(*, count: int = 1) -> Accelerator:
    """NVIDIA L40 - Ada Lovelace professional GPU (2023).

    48GB GDDR6 memory, for visualization and compute.

    Args:
        count: Number of GPUs per node.

    Returns:
        Accelerator specification for L40.
    """
    return Accelerator.from_name("L40", count=count)


def L40S(*, count: int = 1) -> Accelerator:
    """NVIDIA L40S - Ada Lovelace compute GPU (2023).

    48GB GDDR6 memory, optimized for AI/ML workloads.
    Higher power limit than L40 for sustained compute.

    Args:
        count: Number of GPUs per node.

    Returns:
        Accelerator specification for L40S.
    """
    return Accelerator.from_name("L40S", count=count)


# =============================================================================
# NVIDIA Datacenter - Turing/Volta/Pascal (Legacy)
# =============================================================================


def T4(*, count: int = 1) -> Accelerator:
    """NVIDIA T4 - Turing inference GPU (2018).

    16GB GDDR6 memory, excellent cost/performance for inference.
    Widely available on all major clouds (x86_64).

    Args:
        count: Number of GPUs per node.

    Returns:
        Accelerator specification for T4.
    """
    return Accelerator.from_name("T4", count=count)


def T4G(*, count: int = 1) -> Accelerator:
    """NVIDIA T4G - Turing inference GPU for ARM64 (2021).

    16GB GDDR6 memory, same as T4 but for Graviton instances.
    Available on AWS g5g instances.

    Args:
        count: Number of GPUs per node.

    Returns:
        Accelerator specification for T4G.
    """
    return Accelerator.from_name("T4G", count=count)


def V100(
    *,
    memory: Literal["16GB", "32GB"] = "32GB",
    count: int = 1,
) -> Accelerator:
    """NVIDIA V100 - Volta architecture (2017).

    First GPU with Tensor Cores. Available in 16GB and 32GB variants.
    Still widely used for training workloads.

    Args:
        memory: VRAM size - 16GB or 32GB.
        count: Number of GPUs per node.

    Returns:
        Accelerator specification for V100.
    """
    return Accelerator(
        name="V100",
        memory=memory,
        count=count,
        metadata={"cuda": SPECS["V100"]["cuda"]},
    )


def P100(*, count: int = 1) -> Accelerator:
    """NVIDIA P100 - Pascal architecture (2016).

    16GB HBM2 memory, first HBM GPU for deep learning.

    Args:
        count: Number of GPUs per node.

    Returns:
        Accelerator specification for P100.
    """
    return Accelerator.from_name("P100", count=count)


def P40(*, count: int = 1) -> Accelerator:
    """NVIDIA P40 - Pascal inference GPU (2016).

    24GB GDDR5 memory, INT8 inference support.

    Args:
        count: Number of GPUs per node.

    Returns:
        Accelerator specification for P40.
    """
    return Accelerator.from_name("P40", count=count)


def P4(*, count: int = 1) -> Accelerator:
    """NVIDIA P4 - Pascal inference GPU (2016).

    8GB GDDR5 memory, low-power inference.

    Args:
        count: Number of GPUs per node.

    Returns:
        Accelerator specification for P4.
    """
    return Accelerator.from_name("P4", count=count)


def K80(*, count: int = 1) -> Accelerator:
    """NVIDIA K80 - Kepler architecture (2014).

    12GB GDDR5 memory per GPU (24GB total, dual-GPU card).
    Legacy GPU, CUDA support ends at 10.2.

    Args:
        count: Number of GPUs per node.

    Returns:
        Accelerator specification for K80.
    """
    return Accelerator.from_name("K80", count=count)


# =============================================================================
# NVIDIA Consumer RTX - Blackwell (50 series)
# =============================================================================


def RTX_5090(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX 5090 - Blackwell consumer flagship (2025)."""
    return Accelerator.from_name("RTX 5090", count=count)


def RTX_5080(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX 5080 - Blackwell consumer high-end (2025)."""
    return Accelerator.from_name("RTX 5080", count=count)


def RTX_5070_Ti(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX 5070 Ti - Blackwell consumer (2025)."""
    return Accelerator.from_name("RTX 5070 Ti", count=count)


def RTX_5070(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX 5070 - Blackwell consumer (2025)."""
    return Accelerator.from_name("RTX 5070", count=count)


def RTX_5060_Ti(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX 5060 Ti - Blackwell consumer (2025)."""
    return Accelerator.from_name("RTX 5060 Ti", count=count)


def RTX_5060(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX 5060 - Blackwell consumer entry (2025)."""
    return Accelerator.from_name("RTX 5060", count=count)


# =============================================================================
# NVIDIA Consumer RTX - Ada Lovelace (40 series)
# =============================================================================


def RTX_4090(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX 4090 - Ada Lovelace consumer flagship (2022)."""
    return Accelerator.from_name("RTX 4090", count=count)


def RTX_4090D(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX 4090D - China-specific variant."""
    return Accelerator.from_name("RTX 4090D", count=count)


def RTX_4080_Super(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX 4080 Super - Ada Lovelace refresh (2024)."""
    return Accelerator.from_name("RTX 4080 Super", count=count)


def RTX_4080(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX 4080 - Ada Lovelace high-end (2022)."""
    return Accelerator.from_name("RTX 4080", count=count)


def RTX_4070_Ti_Super(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX 4070 Ti Super - Ada Lovelace refresh (2024)."""
    return Accelerator.from_name("RTX 4070 Ti Super", count=count)


def RTX_4070_Ti(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX 4070 Ti - Ada Lovelace (2023)."""
    return Accelerator.from_name("RTX 4070 Ti", count=count)


def RTX_4070_Super(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX 4070 Super - Ada Lovelace refresh (2024)."""
    return Accelerator.from_name("RTX 4070 Super", count=count)


def RTX_4070(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX 4070 - Ada Lovelace (2023)."""
    return Accelerator.from_name("RTX 4070", count=count)


def RTX_4060_Ti(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX 4060 Ti - Ada Lovelace (2023)."""
    return Accelerator.from_name("RTX 4060 Ti", count=count)


def RTX_4060(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX 4060 - Ada Lovelace entry (2023)."""
    return Accelerator.from_name("RTX 4060", count=count)


# =============================================================================
# NVIDIA Consumer RTX - Ampere (30 series)
# =============================================================================


def RTX_3090_Ti(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX 3090 Ti - Ampere flagship (2022)."""
    return Accelerator.from_name("RTX 3090 Ti", count=count)


def RTX_3090(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX 3090 - Ampere flagship (2020)."""
    return Accelerator.from_name("RTX 3090", count=count)


def RTX_3080_Ti(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX 3080 Ti - Ampere high-end (2021)."""
    return Accelerator.from_name("RTX 3080 Ti", count=count)


def RTX_3080(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX 3080 - Ampere high-end (2020)."""
    return Accelerator.from_name("RTX 3080", count=count)


def RTX_3070_Ti(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX 3070 Ti - Ampere (2021)."""
    return Accelerator.from_name("RTX 3070 Ti", count=count)


def RTX_3070(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX 3070 - Ampere (2020)."""
    return Accelerator.from_name("RTX 3070", count=count)


def RTX_3060_Ti(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX 3060 Ti - Ampere (2020)."""
    return Accelerator.from_name("RTX 3060 Ti", count=count)


def RTX_3060(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX 3060 - Ampere (2021)."""
    return Accelerator.from_name("RTX 3060", count=count)


def RTX_3060_Laptop(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX 3060 Laptop - Mobile Ampere."""
    return Accelerator.from_name("RTX 3060 Laptop", count=count)


def RTX_3050(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX 3050 - Ampere entry (2022)."""
    return Accelerator.from_name("RTX 3050", count=count)


# =============================================================================
# NVIDIA Consumer RTX - Turing (20 series)
# =============================================================================


def RTX_2080_Ti(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX 2080 Ti - Turing flagship (2018)."""
    return Accelerator.from_name("RTX 2080 Ti", count=count)


def RTX_2080_Super(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX 2080 Super - Turing refresh (2019)."""
    return Accelerator.from_name("RTX 2080 Super", count=count)


def RTX_2080(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX 2080 - Turing high-end (2018)."""
    return Accelerator.from_name("RTX 2080", count=count)


def RTX_2070_Super(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX 2070 Super - Turing refresh (2019)."""
    return Accelerator.from_name("RTX 2070 Super", count=count)


def RTX_2070(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX 2070 - Turing (2018)."""
    return Accelerator.from_name("RTX 2070", count=count)


def RTX_2060_Super(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX 2060 Super - Turing refresh (2019)."""
    return Accelerator.from_name("RTX 2060 Super", count=count)


def RTX_2060(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX 2060 - Turing entry (2019)."""
    return Accelerator.from_name("RTX 2060", count=count)


# =============================================================================
# NVIDIA Consumer GTX - Turing (16 series)
# =============================================================================


def GTX_1660_Ti(*, count: int = 1) -> Accelerator:
    """NVIDIA GTX 1660 Ti - Turing without RT cores (2019)."""
    return Accelerator.from_name("GTX 1660 Ti", count=count)


def GTX_1660_Super(*, count: int = 1) -> Accelerator:
    """NVIDIA GTX 1660 Super - Turing refresh (2019)."""
    return Accelerator.from_name("GTX 1660 Super", count=count)


def GTX_1660(*, count: int = 1) -> Accelerator:
    """NVIDIA GTX 1660 - Turing without RT cores (2019)."""
    return Accelerator.from_name("GTX 1660", count=count)


# =============================================================================
# NVIDIA Consumer GTX - Pascal (10 series)
# =============================================================================


def GTX_1080_Ti(*, count: int = 1) -> Accelerator:
    """NVIDIA GTX 1080 Ti - Pascal flagship (2017)."""
    return Accelerator.from_name("GTX 1080 Ti", count=count)


def GTX_1080(*, count: int = 1) -> Accelerator:
    """NVIDIA GTX 1080 - Pascal high-end (2016)."""
    return Accelerator.from_name("GTX 1080", count=count)


def GTX_1070_Ti(*, count: int = 1) -> Accelerator:
    """NVIDIA GTX 1070 Ti - Pascal (2017)."""
    return Accelerator.from_name("GTX 1070 Ti", count=count)


def GTX_1070(*, count: int = 1) -> Accelerator:
    """NVIDIA GTX 1070 - Pascal (2016)."""
    return Accelerator.from_name("GTX 1070", count=count)


def GTX_1060(*, count: int = 1) -> Accelerator:
    """NVIDIA GTX 1060 - Pascal entry (2016)."""
    return Accelerator.from_name("GTX 1060", count=count)


def Titan_Xp(*, count: int = 1) -> Accelerator:
    """NVIDIA Titan Xp - Pascal workstation (2017)."""
    return Accelerator.from_name("Titan Xp", count=count)


def Titan_V(*, count: int = 1) -> Accelerator:
    """NVIDIA Titan V - Volta consumer (2017)."""
    return Accelerator.from_name("Titan V", count=count)


# =============================================================================
# NVIDIA Workstation - Ada Lovelace
# =============================================================================


def RTX_6000_Ada(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX 6000 Ada - Workstation flagship (2022)."""
    return Accelerator.from_name("RTX 6000 Ada", count=count)


def RTX_5880_Ada(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX 5880 Ada - Workstation (2023)."""
    return Accelerator.from_name("RTX 5880 Ada", count=count)


def RTX_5000_Ada(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX 5000 Ada - Workstation (2023)."""
    return Accelerator.from_name("RTX 5000 Ada", count=count)


# =============================================================================
# NVIDIA Workstation - Blackwell
# =============================================================================


def RTX_PRO_6000(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX PRO 6000 - Blackwell workstation (2025)."""
    return Accelerator.from_name("RTX PRO 6000", count=count)


def RTX_PRO_4000(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX PRO 4000 - Blackwell workstation (2025)."""
    return Accelerator.from_name("RTX PRO 4000", count=count)


# =============================================================================
# NVIDIA Workstation - Ampere
# =============================================================================


def RTX_A6000(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX A6000 - Ampere workstation flagship (2020)."""
    return Accelerator.from_name("RTX A6000", count=count)


def RTX_A5000(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX A5000 - Ampere workstation (2021)."""
    return Accelerator.from_name("RTX A5000", count=count)


def RTX_A4000(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX A4000 - Ampere workstation (2021)."""
    return Accelerator.from_name("RTX A4000", count=count)


def RTX_A2000(*, count: int = 1) -> Accelerator:
    """NVIDIA RTX A2000 - Ampere workstation entry (2021)."""
    return Accelerator.from_name("RTX A2000", count=count)


# =============================================================================
# NVIDIA Workstation - Turing
# =============================================================================


def Quadro_RTX_8000(*, count: int = 1) -> Accelerator:
    """NVIDIA Quadro RTX 8000 - Turing workstation (2018)."""
    return Accelerator.from_name("Quadro RTX 8000", count=count)


def Quadro_RTX_6000(*, count: int = 1) -> Accelerator:
    """NVIDIA Quadro RTX 6000 - Turing workstation (2018)."""
    return Accelerator.from_name("Quadro RTX 6000", count=count)


def Quadro_RTX_4000(*, count: int = 1) -> Accelerator:
    """NVIDIA Quadro RTX 4000 - Turing workstation (2018)."""
    return Accelerator.from_name("Quadro RTX 4000", count=count)


def Quadro_P4000(*, count: int = 1) -> Accelerator:
    """NVIDIA Quadro P4000 - Pascal workstation (2016)."""
    return Accelerator.from_name("Quadro P4000", count=count)


# =============================================================================
# AMD Instinct
# =============================================================================


def MI300X(*, count: int = 1) -> Accelerator:
    """AMD Instinct MI300X - CDNA 3 flagship (2023).

    192GB HBM3 memory, designed for large language models.
    """
    return Accelerator.from_name("MI300X", count=count)


def MI300B(*, count: int = 1) -> Accelerator:
    """AMD Instinct MI300B - CDNA 3 (2023)."""
    return Accelerator.from_name("MI300B", count=count)


def MI300A(*, count: int = 1) -> Accelerator:
    """AMD Instinct MI300A - CDNA 3 APU (2023).

    Integrated CPU + GPU on single package.
    """
    return Accelerator.from_name("MI300A", count=count)


def MI250X(*, count: int = 1) -> Accelerator:
    """AMD Instinct MI250X - CDNA 2 flagship (2021)."""
    return Accelerator.from_name("MI250X", count=count)


def MI250(*, count: int = 1) -> Accelerator:
    """AMD Instinct MI250 - CDNA 2 (2021)."""
    return Accelerator.from_name("MI250", count=count)


def MI210(*, count: int = 1) -> Accelerator:
    """AMD Instinct MI210 - CDNA 2 (2022)."""
    return Accelerator.from_name("MI210", count=count)


def MI100(*, count: int = 1) -> Accelerator:
    """AMD Instinct MI100 - CDNA 1 (2020)."""
    return Accelerator.from_name("MI100", count=count)


def MI50(*, count: int = 1) -> Accelerator:
    """AMD Instinct MI50 - Vega (2018)."""
    return Accelerator.from_name("MI50", count=count)


# =============================================================================
# AMD RadeonPro
# =============================================================================


def RadeonPro_V710(*, count: int = 1) -> Accelerator:
    """AMD Radeon Pro V710 - Streaming GPU."""
    return Accelerator.from_name("RadeonPro-V710", count=count)


def RadeonPro_V520(*, count: int = 1) -> Accelerator:
    """AMD Radeon Pro V520 - Streaming GPU."""
    return Accelerator.from_name("RadeonPro-V520", count=count)


def Instinct_MI25(*, count: int = 1) -> Accelerator:
    """AMD Instinct MI25 - Vega (2017)."""
    return Accelerator.from_name("Instinct-MI25", count=count)


# =============================================================================
# AWS Trainium
# =============================================================================


def Trainium1(*, count: int = 1) -> Accelerator:
    """AWS Trainium v1 - First-gen training accelerator.

    32GB HBM memory, available on trn1 instances.
    """
    return Accelerator.from_name("Trainium1", count=count)


def Trainium2(*, count: int = 1) -> Accelerator:
    """AWS Trainium v2 - Second-gen training accelerator.

    64GB HBM memory, 4x performance vs Trainium1.
    """
    return Accelerator.from_name("Trainium2", count=count)


def Trainium3(*, count: int = 1) -> Accelerator:
    """AWS Trainium v3 - Third-gen training accelerator.

    128GB HBM memory.
    """
    return Accelerator.from_name("Trainium3", count=count)


# =============================================================================
# AWS Inferentia
# =============================================================================


def Inferentia1(*, count: int = 1) -> Accelerator:
    """AWS Inferentia v1 - First-gen inference accelerator.

    8GB memory, available on inf1 instances.
    """
    return Accelerator.from_name("Inferentia1", count=count)


def Inferentia2(*, count: int = 1) -> Accelerator:
    """AWS Inferentia v2 - Second-gen inference accelerator.

    32GB HBM memory, available on inf2 instances.
    """
    return Accelerator.from_name("Inferentia2", count=count)


# =============================================================================
# Habana Gaudi
# =============================================================================


def Gaudi(*, count: int = 1) -> Accelerator:
    """Habana Gaudi - First-gen training accelerator (2019)."""
    return Accelerator.from_name("Gaudi", count=count)


def Gaudi2(*, count: int = 1) -> Accelerator:
    """Habana Gaudi2 - Second-gen training accelerator (2022).

    96GB HBM2e memory, 2x performance vs Gaudi.
    """
    return Accelerator.from_name("Gaudi2", count=count)


def Gaudi3(*, count: int = 1) -> Accelerator:
    """Habana Gaudi3 - Third-gen training accelerator (2024).

    128GB HBM2e memory.
    """
    return Accelerator.from_name("Gaudi3", count=count)


# =============================================================================
# Google TPU
# =============================================================================


def TPUv2(*, count: int = 1) -> Accelerator:
    """Google TPU v2 - Second-gen tensor processing unit (2017)."""
    return Accelerator.from_name("TPUv2", count=count)


def TPUv3(*, count: int = 1) -> Accelerator:
    """Google TPU v3 - Third-gen TPU (2018)."""
    return Accelerator.from_name("TPUv3", count=count)


def TPUv4(*, count: int = 1) -> Accelerator:
    """Google TPU v4 - Fourth-gen TPU (2021)."""
    return Accelerator.from_name("TPUv4", count=count)


def TPUv5e(*, count: int = 1) -> Accelerator:
    """Google TPU v5e - Efficiency-optimized TPU (2023)."""
    return Accelerator.from_name("TPUv5e", count=count)


def TPUv5p(*, count: int = 1) -> Accelerator:
    """Google TPU v5p - Performance-optimized TPU (2023)."""
    return Accelerator.from_name("TPUv5p", count=count)


def TPUv6(*, count: int = 1) -> Accelerator:
    """Google TPU v6 - Sixth-gen TPU (2024)."""
    return Accelerator.from_name("TPUv6", count=count)


# =============================================================================
# Google TPU Slices
# =============================================================================


def TPUv2_8(*, count: int = 1) -> Accelerator:
    """Google TPU v2-8 - 8-chip slice."""
    return Accelerator.from_name("TPUv2-8", count=count)


def TPUv3_8(*, count: int = 1) -> Accelerator:
    """Google TPU v3-8 - 8-chip slice."""
    return Accelerator.from_name("TPUv3-8", count=count)


def TPUv3_32(*, count: int = 1) -> Accelerator:
    """Google TPU v3-32 - 32-chip slice."""
    return Accelerator.from_name("TPUv3-32", count=count)


def TPUv4_64(*, count: int = 1) -> Accelerator:
    """Google TPU v4-64 - 64-chip slice."""
    return Accelerator.from_name("TPUv4-64", count=count)


def TPUv5e_4(*, count: int = 1) -> Accelerator:
    """Google TPU v5e-4 - 4-chip slice."""
    return Accelerator.from_name("TPUv5e-4", count=count)


def TPUv5p_8(*, count: int = 1) -> Accelerator:
    """Google TPU v5p-8 - 8-chip slice."""
    return Accelerator.from_name("TPUv5p-8", count=count)


# =============================================================================
# Custom Accelerator
# =============================================================================


def Custom(
    name: str,
    *,
    memory: str = "",
    count: int = 1,
    cuda_min: str | None = None,
    cuda_max: str | None = None,
) -> Accelerator:
    """Create a custom accelerator specification.

    Use this for accelerators not in the catalog, or when you need
    to override the default specifications.

    Args:
        name: Accelerator name (e.g., "Custom-GPU", "My-TPU").
        memory: VRAM size (e.g., "48GB").
        count: Number of accelerators per node.
        cuda_min: Minimum CUDA version (e.g., "11.8").
        cuda_max: Maximum CUDA version (e.g., "13.1").

    Returns:
        Accelerator specification with custom parameters.

    Examples:
        >>> Custom("My-GPU", memory="48GB")
        Accelerator(name='My-GPU', memory='48GB')

        >>> Custom("Experimental-TPU", memory="128GB", count=8)
        Accelerator(name='Experimental-TPU', memory='128GB', count=8)

        >>> Custom("H100-Custom", memory="80GB", cuda_min="12.0", cuda_max="13.1")
        Accelerator(name='H100-Custom', memory='80GB', ...)
    """
    metadata = None
    if cuda_min or cuda_max:
        metadata = {"cuda": {}}
        if cuda_min:
            metadata["cuda"]["min"] = cuda_min
        if cuda_max:
            metadata["cuda"]["max"] = cuda_max

    return Accelerator(
        name=name,
        memory=memory,
        count=count,
        metadata=metadata,
    )


