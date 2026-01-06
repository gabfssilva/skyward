"""Accelerator type definitions and detection utilities.

Provides the Accelerator class with nested factory classes for
type-safe accelerator specifications, along with Literal types
for supported accelerators and helper functions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Literal, Union, overload


if TYPE_CHECKING:
    from skyward.cluster import InstanceInfo

# Type for accelerator count: exact number or predicate function
type AcceleratorCount = float | Callable[[int], bool]

# MIG profile types for autocomplete
type H100MIG = Literal[
    "1g.10gb",  # 7 workers, 10GB each
    "1g.20gb",  # 4 workers, 20GB each
    "2g.20gb",  # 3 workers, 20GB each
    "3g.40gb",  # 2 workers, 40GB each
    "4g.40gb",  # 1 worker, 40GB
]

type A100_80GB_MIG = Literal[
    "1g.10gb",  # 7 workers, 10GB each
    "2g.20gb",  # 3 workers, 20GB each
    "3g.40gb",  # 2 workers, 40GB each
    "4g.40gb",  # 1 worker, 40GB
]

type A100_40GB_MIG = Literal[
    "1g.5gb",   # 7 workers, 5GB each
    "2g.10gb",  # 3 workers, 10GB each
    "3g.20gb",  # 2 workers, 20GB each
    "4g.20gb",  # 1 worker, 20GB
]

type A100MIG = A100_40GB_MIG | A100_80GB_MIG


# =============================================================================
# MIG Configuration
# =============================================================================

MIG_CAPABLE: Final[frozenset[str]] = frozenset({
    "H100-80GB", "H100-40GB", "H100-SXM", "H100-PCIe", "H100-NVL",
    "A100-40GB", "A100-80GB", "A100-40", "A100-80",
})

ACCELERATOR_ALIASES: Final[dict[str, str]] = {
    "H100": "H100-80GB",
    "A100": "A100-80GB",
}

# Maximum instances per MIG profile (for homogeneous partitioning)
MIG_MAX_INSTANCES: Final[dict[str, int]] = {
    # H100-80GB / A100-80GB profiles
    "1g.10gb": 7,
    "1g.20gb": 4,
    "2g.20gb": 3,
    "3g.40gb": 2,
    "4g.40gb": 1,
    "7g.80gb": 1,
    # A100-40GB profiles
    "1g.5gb": 7,
    "2g.10gb": 3,
    "3g.20gb": 2,
    "4g.20gb": 1,
    "7g.40gb": 1,
}

# Compute slices used by each profile (GPU has 7 total)
MIG_PROFILE_SLICES: Final[dict[str, int]] = {
    # H100-80GB / A100-80GB profiles
    "1g.10gb": 1,
    "1g.20gb": 1,
    "2g.20gb": 2,
    "3g.40gb": 3,
    "4g.40gb": 4,
    "7g.80gb": 7,
    # A100-40GB profiles
    "1g.5gb": 1,
    "2g.10gb": 2,
    "3g.20gb": 3,
    "4g.20gb": 4,
    "7g.40gb": 7,
}

# Valid profiles per GPU model
MIG_PROFILES_FOR_GPU: Final[dict[str, frozenset[str]]] = {
    "H100-80GB": frozenset({"1g.10gb", "1g.20gb", "2g.20gb", "3g.40gb", "4g.40gb", "7g.80gb"}),
    "H100-40GB": frozenset({"1g.5gb", "2g.10gb", "3g.20gb", "4g.20gb", "7g.40gb"}),
    "H100-SXM": frozenset({"1g.10gb", "1g.20gb", "2g.20gb", "3g.40gb", "4g.40gb", "7g.80gb"}),
    "H100-PCIe": frozenset({"1g.10gb", "1g.20gb", "2g.20gb", "3g.40gb", "4g.40gb", "7g.80gb"}),
    "H100-NVL": frozenset({"1g.10gb", "1g.20gb", "2g.20gb", "3g.40gb", "4g.40gb", "7g.80gb"}),
    "A100-80GB": frozenset({"1g.10gb", "2g.20gb", "3g.40gb", "4g.40gb", "7g.80gb"}),
    "A100-80": frozenset({"1g.10gb", "2g.20gb", "3g.40gb", "4g.40gb", "7g.80gb"}),
    "A100-40GB": frozenset({"1g.5gb", "2g.10gb", "3g.20gb", "4g.20gb", "7g.40gb"}),
    "A100-40": frozenset({"1g.5gb", "2g.10gb", "3g.20gb", "4g.20gb", "7g.40gb"}),
}


def _normalize_accelerator(acc: str) -> str:
    return ACCELERATOR_ALIASES.get(acc, acc)


@dataclass(frozen=True)
class Accelerator:
    """Accelerator specification with factory methods via nested classes.

    Examples:
        >>> Accelerator.NVIDIA.H100()           # 1 GPU full
        >>> Accelerator.NVIDIA.H100(count=4)    # 4 GPUs
        >>> Accelerator.NVIDIA.H100(count=lambda c: c >= 2)  # at least 2 GPUs
        >>> Accelerator.NVIDIA.H100(mig="3g.40gb")  # 1 MIG partition
        >>> Accelerator.AMD.MI("300X")          # 1 MI300X
    """

    accelerator: str
    memory: str
    count: AcceleratorCount = 1
    multiple_instance: str | list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_name(cls, name: str) -> "Accelerator | None":
        """Create Accelerator from GPU name string (dynamic factory lookup)."""
        for vendor in (cls.NVIDIA, cls.AMD, cls.Habana, cls.AWS, cls.Google):
            if hasattr(vendor, name):
                return getattr(vendor, name)()
        return None

    @classmethod
    def from_value(cls, value: "Accelerator | str | None") -> "Accelerator | None":
        """Create Accelerator from various input types."""
        if value is None:
            return None
        if isinstance(value, Accelerator):
            return value
        return cls(accelerator=value, memory="", count=1)

    @property
    def fractional(self) -> bool:
        """True if using fractional GPU (0 < count < 1)."""
        if callable(self.count):
            return False
        return 0 < self.count < 1

    class NVIDIA:
        @staticmethod
        def H100(
            count: AcceleratorCount = 1,
            memory: Literal["40GB", "80GB"] = "80GB",
            form_factor: Literal["SXM", "PCIe", "NVL"] | None = None,
            mig: H100MIG | list[H100MIG] | None = None,
        ) -> "Accelerator":
            if form_factor:
                name = f"H100-{form_factor}"
                mem = "94GB" if form_factor == "NVL" else memory
            else:
                name = "H100"
                mem = memory
            return Accelerator(name, mem, count, mig, {"cuda": {"min": "11.8", "max": "13.1"}})

        @staticmethod
        def H200(count: AcceleratorCount = 1) -> "Accelerator":
            return Accelerator("H200", "141GB", count, metadata={"cuda": {"min": "11.8", "max": "13.1"}})

        @staticmethod
        @overload
        def A100(
            count: AcceleratorCount = 1,
            memory: Literal["80GB"] = "80GB",
            mig: A100_80GB_MIG | list[A100_80GB_MIG] | None = None,
        ) -> "Accelerator": ...

        @staticmethod
        @overload
        def A100(
            count: AcceleratorCount = 1,
            memory: Literal["40GB"] = "40GB",
            mig: A100_40GB_MIG | list[A100_40GB_MIG] | None = None,
        ) -> "Accelerator": ...

        @staticmethod
        def A100(
            count: AcceleratorCount = 1,
            memory: Literal["40GB", "80GB"] = "80GB",
            mig: A100MIG | list[A100MIG] | None = None,
        ) -> "Accelerator":
            return Accelerator("A100", memory, count, mig, {"cuda": {"min": "11.0", "max": "13.1"}})

        @staticmethod
        def B100(count: AcceleratorCount = 1) -> "Accelerator":
            return Accelerator("B100", "192GB", count, metadata={"cuda": {"min": "12.8", "max": "13.1"}})

        @staticmethod
        def B200(count: AcceleratorCount = 1) -> "Accelerator":
            return Accelerator("B200", "192GB", count, metadata={"cuda": {"min": "12.8", "max": "13.1"}})

        @staticmethod
        def GB200(count: AcceleratorCount = 1) -> "Accelerator":
            return Accelerator("GB200", "384GB", count, metadata={"cuda": {"min": "12.8", "max": "13.1"}})

        @staticmethod
        def GH200(count: AcceleratorCount = 1) -> "Accelerator":
            return Accelerator("GH200", "96GB", count, metadata={"cuda": {"min": "11.8", "max": "13.1"}})

        @staticmethod
        def L4(count: AcceleratorCount = 1) -> "Accelerator":
            return Accelerator("L4", "24GB", count, metadata={"cuda": {"min": "11.8", "max": "13.1"}})

        @staticmethod
        def L40(count: AcceleratorCount = 1) -> "Accelerator":
            return Accelerator("L40", "48GB", count, metadata={"cuda": {"min": "11.8", "max": "13.1"}})

        @staticmethod
        def L40S(count: AcceleratorCount = 1) -> "Accelerator":
            return Accelerator("L40S", "48GB", count, metadata={"cuda": {"min": "11.8", "max": "13.1"}})

        @staticmethod
        def T4(count: AcceleratorCount = 1) -> "Accelerator":
            return Accelerator("T4", "16GB", count, metadata={"cuda": {"min": "10.0", "max": "13.1"}})

        @staticmethod
        def A10(count: AcceleratorCount = 1) -> "Accelerator":
            return Accelerator("A10", "24GB", count, metadata={"cuda": {"min": "11.0", "max": "13.1"}})

        @staticmethod
        def A10G(count: AcceleratorCount = 1) -> "Accelerator":
            return Accelerator("A10G", "24GB", count, metadata={"cuda": {"min": "11.0", "max": "13.1"}})

        @staticmethod
        def A2(count: AcceleratorCount = 1) -> "Accelerator":
            return Accelerator("A2", "16GB", count, metadata={"cuda": {"min": "11.0", "max": "13.1"}})

        @staticmethod
        def V100(count: AcceleratorCount = 1, memory: Literal["16GB", "32GB"] = "32GB") -> "Accelerator":
            return Accelerator("V100", memory, count, metadata={"cuda": {"min": "9.0", "max": "12.6"}})

        @staticmethod
        def P100(count: AcceleratorCount = 1) -> "Accelerator":
            return Accelerator("P100", "16GB", count, metadata={"cuda": {"min": "8.0", "max": "12.6"}})

        @staticmethod
        def P4(count: AcceleratorCount = 1) -> "Accelerator":
            return Accelerator("P4", "8GB", count, metadata={"cuda": {"min": "8.0", "max": "12.6"}})

        @staticmethod
        def K80(count: AcceleratorCount = 1) -> "Accelerator":
            return Accelerator("K80", "12GB", count, metadata={"cuda": {"min": "5.5", "max": "10.2"}})

        @staticmethod
        def RTX3080(count: AcceleratorCount = 1) -> "Accelerator":
            return Accelerator("RTX 3080", "10GB", count, metadata={"cuda": {"min": "11.1", "max": "13.1"}})

        @staticmethod
        def RTX3090(count: AcceleratorCount = 1) -> "Accelerator":
            return Accelerator("RTX 3090", "24GB", count, metadata={"cuda": {"min": "11.1", "max": "13.1"}})

        @staticmethod
        def RTX3080Ti(count: AcceleratorCount = 1) -> "Accelerator":
            return Accelerator("RTX 3080 Ti", "12GB", count, metadata={"cuda": {"min": "11.1", "max": "13.1"}})

        @staticmethod
        def RTX4080(count: AcceleratorCount = 1) -> "Accelerator":
            return Accelerator("RTX 4080", "16GB", count, metadata={"cuda": {"min": "11.8", "max": "13.1"}})

        @staticmethod
        def RTX4090(count: AcceleratorCount = 1) -> "Accelerator":
            return Accelerator("RTX 4090", "24GB", count, metadata={"cuda": {"min": "11.8", "max": "13.1"}})

        @staticmethod
        def RTX5090(count: AcceleratorCount = 1) -> "Accelerator":
            return Accelerator("RTX 5090", "32GB", count, metadata={"cuda": {"min": "12.8", "max": "13.1"}})

    class AMD:
        @staticmethod
        def MI(
            model: Literal["50", "100", "210", "250", "250X", "300A", "300B", "300X"],
            count: AcceleratorCount = 1,
        ) -> "Accelerator":
            memory = {
                "50": "16GB", "100": "32GB", "210": "64GB",
                "250": "128GB", "250X": "128GB",
                "300A": "128GB", "300B": "192GB", "300X": "192GB",
            }
            return Accelerator(f"MI{model}", memory[model], count)

        @staticmethod
        def RadeonPro(
            model: Literal["V520", "V710"],
            count: AcceleratorCount = 1,
        ) -> "Accelerator":
            """Radeon Pro GPUs for graphics/streaming workloads (AWS g4ad, Azure)."""
            memory = {"V520": "8GB", "V710": "16GB"}
            return Accelerator(f"RadeonPro-{model}", memory[model], count)

        @staticmethod
        def Instinct(
            model: Literal["MI25"],
            count: AcceleratorCount = 1,
        ) -> "Accelerator":
            """Radeon Instinct GPUs (Azure NVv4)."""
            memory = {"MI25": "16GB"}
            return Accelerator(f"Instinct-{model}", memory[model], count)

    class Habana:
        @staticmethod
        def Gaudi(version: Literal[1, 2, 3] = 3, count: AcceleratorCount = 1) -> "Accelerator":
            memory = {1: "32GB", 2: "96GB", 3: "128GB"}
            name = "Gaudi" if version == 1 else f"Gaudi{version}"
            return Accelerator(name, memory[version], count)

    class AWS:
        @staticmethod
        def Trainium(version: Literal[1, 2, 3] = 2, count: AcceleratorCount = 1) -> "Accelerator":
            memory = {1: "32GB", 2: "64GB", 3: "128GB"}
            return Accelerator(f"Trainium{version}", memory[version], count)

        @staticmethod
        def Inferentia(version: Literal[1, 2] = 2, count: AcceleratorCount = 1) -> "Accelerator":
            memory = {1: "8GB", 2: "32GB"}
            return Accelerator(f"Inferentia{version}", memory[version], count)

    class Google:
        @staticmethod
        def TPU(
            version: Literal["v2", "v3", "v4", "v5e", "v5p", "v6"] = "v5p",
            count: AcceleratorCount = 1,
        ) -> "Accelerator":
            memory = {"v2": "8GB", "v3": "16GB", "v4": "32GB", "v5e": "16GB", "v5p": "95GB", "v6": "32GB"}
            return Accelerator(f"TPU{version}", memory[version], count)

        @staticmethod
        def TPUSlice(
            version: Literal["v2-8", "v3-8", "v3-32", "v4-64", "v5e-4", "v5p-8"],
        ) -> "Accelerator":
            config = {
                "v2-8": ("64GB", 8),
                "v3-8": ("128GB", 8),
                "v3-32": ("512GB", 32),
                "v4-64": ("2TB", 64),
                "v5e-4": ("64GB", 4),
                "v5p-8": ("760GB", 8),
            }
            mem, cnt = config[version]
            return Accelerator(f"TPU{version}", mem, cnt)


# =============================================================================
# NVIDIA Literal Types
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

GPU = Union[
    NVIDIA,
    AMD,
    None,
]

AcceleratorType = Union[
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
    "Instinct-MI25", "RadeonPro-V520", "RadeonPro-V710",
})

_HABANA_VALUES: Final[frozenset[str]] = frozenset({
    "Gaudi", "Gaudi2", "Gaudi3",
})


# =============================================================================
# Type Checking Functions
# =============================================================================


def is_nvidia(acc: AcceleratorType) -> bool:
    """Check if accelerator is an NVIDIA GPU."""
    return acc in _NVIDIA_VALUES


def is_trainium(acc: AcceleratorType) -> bool:
    """Check if accelerator is AWS Trainium."""
    return acc in _TRAINIUM_VALUES


def is_gpu(acc: AcceleratorType) -> bool:
    """Check if accelerator is any GPU (NVIDIA or AMD)."""
    return acc in _NVIDIA_VALUES or acc in _AMD_VALUES


def current_accelerator() -> AcceleratorType:
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
