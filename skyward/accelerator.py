"""Accelerator type definitions and detection utilities.

Provides type-safe Literal types for supported accelerators,
AcceleratorSpec for worker isolation configurations, and
factory functions with autocomplete support.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, Union

if TYPE_CHECKING:
    from skyward.cluster import InstanceInfo
#
# class Acc(ABC):
#     accelerator: str
#     memory: str
#     count: int = 1
#     type: Literal['gpu', 'npu', 'tpu'] = 'gpu'
#
# @dataclass(frozen=True)
# class NVIDIA(Acc):
#     accelerator: NVIDIA
#     memory: str
#     count: int = 1
#     type: Literal['gpu', 'npu', 'tpu'] = 'gpu'
#     mig: str | list[str] | None = None
#
# @dataclass(frozen=True)
# class H100(NVIDIA):
#     accelerator: Literal['H100']
#     count: int = 1
#     memory: Literal['40GB', '80GB'] = '40GB'
#     mig: MultiInstance
#
#     type MultiInstance = Literal[
#         "1g.10gb",  # 7 workers, 10GB each
#         "1g.20gb",  # 4 workers, 20GB each
#         "2g.20gb",  # 3 workers, 20GB each
#         "3g.40gb",  # 2 workers, 40GB each
#         "4g.40gb",  # 1 worker, 40GB
#     ]

# =============================================================================
# Type Aliases for Factory Functions
# =============================================================================

type AcceleratorCount = Literal[1, 2, 3, 4, 5, 6, 7, 8]

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


# =============================================================================
# AcceleratorSpec
# =============================================================================


@dataclass(frozen=True, slots=True)
class AcceleratorSpec:
    """Accelerator specification with MIG partitioning support.

    Attributes:
        accelerator: Accelerator type (e.g., "H100-80GB", "T4").
        count: Number of GPUs.
        mig: MIG profile(s) for partitioning. None = no MIG (full GPU).
             String = homogeneous (e.g., "3g.40gb" → 2 workers per GPU).
             Tuple = heterogeneous (e.g., ("4g.40gb", "3g.40gb") → 2 workers).

    Examples:
        >>> H100()                           # 1 GPU, 1 worker, 80GB
        >>> H100(count=4)                    # 4 GPUs, 4 workers
        >>> H100(mig="3g.40gb")              # 1 GPU, 2 workers de 40GB
        >>> H100(count=2, mig="3g.40gb")     # 2 GPUs, 4 workers
        >>> H100(mig=("4g.40gb", "3g.40gb")) # 1 GPU, 2 workers (heterogeneous)
    """

    accelerator: str
    count: int = 1
    mig: str | tuple[str, ...] | None = None

    @property
    def workers(self) -> int:
        """Total number of workers across all GPUs.

        Without MIG: 1 worker manages all GPUs (multi-GPU data parallel).
        With MIG: 1 worker per MIG partition (each partition is isolated).
        """
        if self.mig is None:
            return 1  # Single process can manage multiple GPUs
        if isinstance(self.mig, tuple):
            return self.count * len(self.mig)
        return self.count * MIG_MAX_INSTANCES[self.mig]

    @property
    def workers_per_gpu(self) -> int:
        """Number of workers per GPU."""
        if self.mig is None:
            return 1
        if isinstance(self.mig, tuple):
            return len(self.mig)
        return MIG_MAX_INSTANCES[self.mig]

    @property
    def requires_mig(self) -> bool:
        """True if MIG partitioning is needed."""
        return self.mig is not None

    @property
    def is_heterogeneous(self) -> bool:
        """True if using different MIG profiles per GPU."""
        return isinstance(self.mig, tuple) and len(set(self.mig)) > 1

    @property
    def mig_profiles(self) -> tuple[str, ...] | None:
        """MIG profiles as tuple (for script generation)."""
        if self.mig is None:
            return None
        if isinstance(self.mig, str):
            return (self.mig,) * MIG_MAX_INSTANCES[self.mig]
        return self.mig

    def validate(self) -> None:
        if self.count <= 0:
            raise ValueError(f"count must be positive, got {self.count}")

        if self.mig is not None:
            normalized = _normalize_accelerator(self.accelerator)
            if normalized not in MIG_CAPABLE:
                raise ValueError(
                    f"{self.accelerator} does not support MIG partitioning"
                )

            # Get valid profiles for this GPU
            valid_profiles = MIG_PROFILES_FOR_GPU.get(normalized, set())

            # Validate all profiles
            profiles = (self.mig,) if isinstance(self.mig, str) else self.mig
            for profile in profiles:
                if profile not in valid_profiles:
                    raise ValueError(
                        f"Profile '{profile}' not valid for {self.accelerator}. "
                        f"Valid profiles: {sorted(valid_profiles)}"
                    )

            # For heterogeneous, validate total slices <= 7
            if isinstance(self.mig, tuple):
                total_slices = sum(MIG_PROFILE_SLICES[p] for p in self.mig)
                if total_slices > 7:
                    raise ValueError(
                        f"MIG profiles {self.mig} use {total_slices} slices, "
                        f"but GPU only has 7 slices"
                    )

    @classmethod
    def from_value(
        cls, value: AcceleratorSpec | str | None
    ) -> AcceleratorSpec | None:
        """Create AcceleratorSpec from various input types.

        Args:
            value: AcceleratorSpec, Accelerator string, or None.

        Returns:
            AcceleratorSpec or None if value is None.
        """
        if value is None:
            return None
        if isinstance(value, AcceleratorSpec):
            return value
        # Accelerator literal (string like "H100")
        return cls(accelerator=value, count=1, mig=None)


# =============================================================================
# Factory Functions (MIG-capable accelerators)
# =============================================================================


def H100(
    memory: Literal["40", "80"] = "80",
    count: AcceleratorCount = 1,
    mig: H100MIG | list[H100MIG] | None = None,
) -> AcceleratorSpec:
    """Create H100 GPU specification.

    Args:
        memory: VRAM variant ("40" or "80" GB).
        count: Number of GPUs.
        mig: MIG profile(s) for partitioning.
            - None: No MIG, 1 worker per GPU (full 80GB)
            - "3g.40gb": 2 workers per GPU, 40GB each
            - "2g.20gb": 3 workers per GPU, 20GB each
            - "1g.10gb": 7 workers per GPU, 10GB each
            - ("4g.40gb", "3g.40gb"): heterogeneous, 2 workers using 7/7 slices

    Examples:
        >>> H100()                           # 1 GPU, 1 worker, 80GB
        >>> H100(count=4)                    # 4 GPUs, 4 workers
        >>> H100(mig="3g.40gb")              # 1 GPU, 2 workers de 40GB
        >>> H100(mig=("4g.40gb", "3g.40gb")) # 1 GPU, 2 workers (7/7 slices)
    """
    return AcceleratorSpec(f"H100-{memory}GB", count, tuple(mig))


def A100(
    memory: Literal["40", "80"] = "80",
    count: AcceleratorCount = 1,
    mig: A100_80GB_MIG | A100_40GB_MIG | tuple[str, ...] | None = None,
) -> AcceleratorSpec:
    """Create A100 GPU specification.

    Args:
        memory: VRAM variant ("40" or "80" GB).
        count: Number of GPUs.
        mig: MIG profile(s) for partitioning.

    Examples:
        >>> A100()                  # 1 GPU, 1 worker
        >>> A100(mig="3g.40gb")     # 1 GPU, 2 workers de 40GB (80GB variant)
        >>> A100(memory="40", mig="3g.20gb")  # 1 GPU, 2 workers de 20GB
    """
    return AcceleratorSpec(f"A100-{memory}GB", count, mig)


# =============================================================================
# Factory Functions (Non-MIG accelerators)
# =============================================================================


def T4(count: AcceleratorCount = 1) -> AcceleratorSpec:
    """Create T4 GPU specification (no MIG support)."""
    return AcceleratorSpec("T4", count)


def L4(count: AcceleratorCount = 1) -> AcceleratorSpec:
    """Create L4 GPU specification (no MIG support)."""
    return AcceleratorSpec("L4", count)

def M60(count: AcceleratorCount = 1) -> AcceleratorSpec:
    return AcceleratorSpec("M60", count)

def L40(count: AcceleratorCount = 1) -> AcceleratorSpec:
    """Create L40 GPU specification (no MIG support)."""
    return AcceleratorSpec("L40", count)


def L40S(count: AcceleratorCount = 1) -> AcceleratorSpec:
    """Create L40S GPU specification (no MIG support)."""
    return AcceleratorSpec("L40S", count)


def V100(count: AcceleratorCount = 1) -> AcceleratorSpec:
    """Create V100 GPU specification (no MIG support)."""
    return AcceleratorSpec("V100", count)


def P100(count: AcceleratorCount = 1) -> AcceleratorSpec:
    """Create P100 GPU specification (no MIG support)."""
    return AcceleratorSpec("P100", count)


def RTX3090(count: AcceleratorCount = 1) -> AcceleratorSpec:
    """Create RTX 3090 GPU specification (no MIG support)."""
    return AcceleratorSpec("RTX3090", count)


def RTX4090(count: AcceleratorCount = 1) -> AcceleratorSpec:
    """Create RTX 4090 GPU specification (no MIG support)."""
    return AcceleratorSpec("RTX4090", count)


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
