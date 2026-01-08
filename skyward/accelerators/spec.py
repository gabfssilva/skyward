"""AcceleratorSpec dataclass and Accelerator factory function.

Provides type-safe accelerator specifications with overloaded factory
for IDE autocomplete support.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, overload

from .catalog import SPECS
from .types import NVIDIA, TPU, Trainium

if TYPE_CHECKING:
    from skyward.cluster.info import InstanceInfo

# Type for accelerator count: exact number or predicate function
type AcceleratorCount = float | Callable[[int], bool]


@dataclass(frozen=True)
class AcceleratorSpec:
    """Accelerator specification with factory methods.

    Examples:
        >>> Accelerator('T4')
        >>> Accelerator('H100', memory='80GB', count=4)
        >>> Accelerator('A100', count=lambda c: c >= 2)
    """

    accelerator: str
    memory: str
    count: AcceleratorCount = 1
    multiple_instance: str | list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_name(cls, name: str) -> AcceleratorSpec | None:
        """Create Accelerator from GPU name string."""
        if name in SPECS:
            return Accelerator(name)
        return None

    @classmethod
    def from_value(cls, value: AcceleratorSpec | str | None) -> AcceleratorSpec | None:
        """Create Accelerator from various input types."""
        if value is None:
            return None
        if isinstance(value, AcceleratorSpec):
            return value
        return cls(accelerator=value, memory="", count=1)

    @property
    def fractional(self) -> bool:
        """True if using fractional GPU (0 < count < 1)."""
        if callable(self.count):
            return False
        return 0 < self.count < 1


def current_accelerator() -> NVIDIA | Trainium | TPU | None:
    """Get the accelerator type for the current compute pool.

    Returns:
        Accelerator type if running in a compute pool, None otherwise.
    """
    from skyward.cluster.info import instance_info

    pool: InstanceInfo | None = instance_info()
    if pool is None:
        return None
    acc = pool.accelerator
    if acc is None:
        return None
    return acc.get("type")


# =============================================================================
# Accelerator() Overloads
# =============================================================================

# NVIDIA - single memory option
@overload
def Accelerator(name: Literal[
    # Datacenter - Legacy
    "K80", "P4", "P40", "P100",
    # Datacenter - Turing/Ampere
    "T4", "A2", "A10", "A10G", "A40", "A800",
    # Datacenter - Ada Lovelace
    "L4", "L40", "L40S",
    # Datacenter - Hopper
    "H200", "H200-NVL", "GH200",
    # Datacenter - Blackwell
    "B100", "B200", "GB200",
    # Consumer GTX - Pascal
    "GTX 1060", "GTX 1070", "GTX 1070 Ti", "GTX 1080", "GTX 1080 Ti", "Titan Xp",
    # Consumer GTX - Turing
    "GTX 1660", "GTX 1660 Super", "GTX 1660 Ti",
    # Consumer RTX - Turing (20 series)
    "RTX 2060", "RTX 2060 Super", "RTX 2070", "RTX 2070 Super",
    "RTX 2080", "RTX 2080 Super", "RTX 2080 Ti",
    # Consumer RTX - Ampere (30 series)
    "RTX 3050", "RTX 3060", "RTX 3060 Ti", "RTX 3060 Laptop", "RTX 3070", "RTX 3070 Ti",
    "RTX 3080", "RTX 3080 Ti", "RTX 3090", "RTX 3090 Ti",
    # Consumer RTX - Ada Lovelace (40 series)
    "RTX 4060", "RTX 4060 Ti", "RTX 4070", "RTX 4070 Super",
    "RTX 4070 Ti", "RTX 4070 Ti Super", "RTX 4080", "RTX 4080 Super", "RTX 4090", "RTX 4090D",
    # Consumer RTX - Blackwell (50 series)
    "RTX 5060", "RTX 5060 Ti", "RTX 5070", "RTX 5070 Ti", "RTX 5080", "RTX 5090",
    # Workstation - Pascal/Volta
    "Quadro P4000", "Titan V",
    # Workstation - Turing
    "Quadro RTX 4000", "Quadro RTX 6000", "Quadro RTX 8000",
    # Workstation - Ampere
    "RTX A2000", "RTX A4000", "RTX A5000", "RTX A6000",
    # Workstation - Ada Lovelace
    "RTX 5000 Ada", "RTX 5880 Ada", "RTX 6000 Ada",
    # Workstation - Blackwell
    "RTX PRO 4000", "RTX PRO 6000",
], *, count: AcceleratorCount = 1) -> AcceleratorSpec: ...


# NVIDIA - multiple memory options
@overload
def Accelerator(
    name: Literal["H100"],
    *,
    memory: Literal["40GB", "80GB"] = ...,
    form_factor: Literal["SXM", "PCIe", "NVL"] | None = ...,
    count: AcceleratorCount = 1,
) -> AcceleratorSpec: ...


@overload
def Accelerator(
    name: Literal["A100"],
    *,
    memory: Literal["40GB", "80GB"] = ...,
    count: AcceleratorCount = 1,
) -> AcceleratorSpec: ...


@overload
def Accelerator(
    name: Literal["V100"],
    *,
    memory: Literal["16GB", "32GB"] = ...,
    count: AcceleratorCount = 1,
) -> AcceleratorSpec: ...


# AMD
@overload
def Accelerator(
    name: Literal["MI50", "MI100", "MI210", "MI250", "MI250X", "MI300A", "MI300B", "MI300X"],
    *,
    count: AcceleratorCount = 1,
) -> AcceleratorSpec: ...


@overload
def Accelerator(
    name: Literal["RadeonPro-V520", "RadeonPro-V710"],
    *,
    count: AcceleratorCount = 1,
) -> AcceleratorSpec: ...


@overload
def Accelerator(
    name: Literal["Instinct-MI25"],
    *,
    count: AcceleratorCount = 1,
) -> AcceleratorSpec: ...


# AWS
@overload
def Accelerator(
    name: Literal["Trainium"],
    *,
    version: Literal[1, 2, 3] = ...,
    count: AcceleratorCount = 1,
) -> AcceleratorSpec: ...


@overload
def Accelerator(
    name: Literal["Inferentia"],
    *,
    version: Literal[1, 2] = ...,
    count: AcceleratorCount = 1,
) -> AcceleratorSpec: ...


# Habana
@overload
def Accelerator(
    name: Literal["Gaudi"],
    *,
    version: Literal[1, 2, 3] = ...,
    count: AcceleratorCount = 1,
) -> AcceleratorSpec: ...


# Google TPU
@overload
def Accelerator(
    name: Literal["TPU"],
    *,
    version: Literal["v2", "v3", "v4", "v5e", "v5p", "v6"] = ...,
    count: AcceleratorCount = 1,
) -> AcceleratorSpec: ...


@overload
def Accelerator(
    name: Literal["TPUSlice"],
    *,
    slice: Literal["v2-8", "v3-8", "v3-32", "v4-64", "v5e-4", "v5p-8"],
) -> AcceleratorSpec: ...


def Accelerator(
    name: str,
    *,
    memory: str | None = None,
    version: int | str | None = None,
    form_factor: str | None = None,
    slice: str | None = None,
    count: AcceleratorCount = 1,
) -> AcceleratorSpec:
    """Create an Accelerator specification.

    Examples:
        >>> Accelerator('T4')
        >>> Accelerator('H100', memory='80GB', count=4)
        >>> Accelerator('A100', count=lambda c: c >= 2)
        >>> Accelerator('Trainium', version=2)
        >>> Accelerator('TPU', version='v5p')
        >>> Accelerator('TPUSlice', slice='v5p-8')
    """
    # Resolve versioned names
    match name:
        case "Trainium":
            resolved = f"Trainium{version or 2}"
        case "Inferentia":
            resolved = f"Inferentia{version or 2}"
        case "Gaudi":
            v = version or 3
            resolved = "Gaudi" if v == 1 else f"Gaudi{v}"
        case "TPU":
            resolved = f"TPU{version or 'v5p'}"
        case "TPUSlice":
            if slice is None:
                raise ValueError("TPUSlice requires 'slice' parameter")
            resolved = f"TPU{slice}"
        case _:
            resolved = name

    spec = SPECS.get(resolved)
    if spec is None:
        raise ValueError(f"Unknown accelerator: {name}")

    # Handle H100 form factor
    if name == "H100" and form_factor:
        resolved = f"H100-{form_factor}"
        mem = "94GB" if form_factor == "NVL" else (memory or "80GB")
    else:
        mem = memory or spec["memory"]

    # TPU slices have fixed count
    final_count = spec.get("count", count)

    metadata = {"cuda": spec["cuda"]} if "cuda" in spec else {}

    return AcceleratorSpec(
        accelerator=resolved,
        memory=mem,
        count=final_count,
        metadata=metadata,
    )
