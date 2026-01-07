"""Type definitions for skyward using Python 3.12+ generics."""

from skyward.accelerators import (
    GPU,
    NVIDIA,
    AcceleratorCount,
    AcceleratorSpec,
    Trainium,
    current_accelerator,
)
from skyward.types.core import (
    Architecture,
    Auto,
    Megabytes,
    Memory,
)
from skyward.types.instance import (
    ExitedInstance,
    Instance,
)
from skyward.types.protocols import (
    ComputeSpec,
    Instances,
    Provider,
    ProviderConfig,
)
from skyward.types.selection import (
    ProviderLike,
    ProviderLiteral,
    ProviderSelector,
    SelectionLike,
    SelectionStrategy,
    SingleProvider,
)
from skyward.types.spec import (
    InstanceSpec,
    parse_memory_mb,
    select_instance,
    select_instances,
)

__all__ = [
    # Accelerators (re-exported from skyward.accelerators)
    "NVIDIA",
    "Trainium",
    "GPU",
    "AcceleratorSpec",
    "AcceleratorCount",
    "current_accelerator",
    # Core types
    "Auto",
    "Architecture",
    "Megabytes",
    "Memory",
    # Instance specification
    "InstanceSpec",
    "select_instance",
    "select_instances",
    "parse_memory_mb",
    # Instance
    "Instance",
    "ExitedInstance",
    # Protocols
    "Instances",
    "ComputeSpec",
    "Provider",
    "ProviderConfig",
    # Provider selection
    "ProviderLiteral",
    "SelectionStrategy",
    "ProviderSelector",
    "SelectionLike",
    "SingleProvider",
    "ProviderLike",
]
