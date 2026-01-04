"""Skyward - Execute Python functions on cloud compute.

Example (explicit pool):

    import skyward as sky

    @sky.compute
    def train(data):
        return model.fit(data)

    pool = sky.ComputePool(
        provider=sky.AWS(),
        accelerator=sky.Accelerator.NVIDIA.A100(),
        image=sky.Image(pip=["torch"]),
    )

    with pool:
        result = train(data) >> pool
        r1, r2 = sky.gather(train(d1), train(d2)) >> pool

Example (implicit pool with decorator):

    import skyward as sky

    @sky.compute
    def train(data):
        return model.fit(data)

    @sky.pool(
        provider=sky.AWS(),
        accelerator=sky.Accelerator.NVIDIA.A100(),
        image=sky.Image(pip=["torch"]),
    )
    def main():
        result = train(data) >> sky
        r1, r2 = sky.gather(train(d1), train(d2)) >> sky
        return result

    main()  # provisions -> executes -> deprovisions
"""

# Accelerator utilities
import skyward.conc as conc
import skyward.integrations as integrations

# Pool decorator (implicit context)
from skyward._pool_decorator import pool
from skyward.accelerator import (
    is_nvidia,
    is_trainium,
)

# Callback system
from skyward.callback import Callback, compose, emit, use_callback

# Cluster utilities
from skyward.cluster import InstanceInfo, instance_info

# Data sharding utilities
from skyward.data import DistributedSampler, shard, shard_iterator

# Events (ADT)
from skyward.events import (
    BootstrapCompleted,
    BootstrapProgress,
    BootstrapStarting,
    CostFinal,
    CostUpdate,
    Error,
    InfraCreated,
    InfraCreating,
    InstanceLaunching,
    InstanceProvisioned,
    InstanceStopping,
    LogLine,
    Metrics,
    PoolStarted,
    PoolStopping,
    SkywardEvent,
)

# Executor
from skyward.executor import Executor

# Image
from skyward.image import Image

# Logging configuration
from skyward.logging import LogConfig
from skyward.multi_pool import MultiPool

# Lazy computation API
from skyward.pending import (
    ComputeFunction,
    PendingBatch,
    PendingCompute,
    compute,
    gather,
    lazy,
)

# Pool
from skyward.pool import ComputePool

# Providers
from skyward.providers import AWS, DigitalOcean, Verda

# Provider selection
from skyward.selection import (
    AllProvidersFailedError,
    NoAvailableProviderError,
    select_available,
    select_cheapest,
    select_first,
)

# Allocation strategies
from skyward.spec import Allocation, AllocationLike

# Types
from skyward.types import (
    GPU,
    NVIDIA,
    Accelerator,
    Architecture,
    Auto,
    ComputeSpec,
    ExitedInstance,
    Instance,
    InstanceSpec,
    Provider,
    ProviderConfig,
    ProviderLike,
    ProviderSelector,
    SelectionLike,
    SelectionStrategy,
    Trainium,
    current_accelerator,
    select_instance,
)

# Volumes
from skyward.volume import S3Volume, Volume

__version__ = "0.2.0"

__all__ = [
    # === Lazy API ===
    # Lazy computation
    "compute",
    "lazy",  # Alias for compute
    "gather",
    "PendingCompute",
    "PendingBatch",
    "ComputeFunction",
    # Pool
    "ComputePool",
    "MultiPool",
    "Executor",
    "pool",  # Decorator for implicit context
    # Image
    "Image",
    # Callback system
    "emit",
    "use_callback",
    "compose",
    "Callback",
    # Logging
    "LogConfig",
    # === Types ===
    "Instance",
    "InstanceSpec",
    "ExitedInstance",
    "Provider",
    "ProviderConfig",
    "ComputeSpec",
    "Architecture",
    "Auto",
    "select_instance",
    # Allocation strategies
    "Allocation",
    "AllocationLike",
    # Events (ADT)
    "SkywardEvent",
    "InfraCreating",
    "InfraCreated",
    "InstanceLaunching",
    "InstanceProvisioned",
    "BootstrapStarting",
    "BootstrapProgress",
    "BootstrapCompleted",
    "Metrics",
    "LogLine",
    "InstanceStopping",
    "CostUpdate",
    "CostFinal",
    "PoolStarted",
    "PoolStopping",
    "Error",
    # Accelerators
    "Accelerator",
    "GPU",
    "NVIDIA",
    "Trainium",
    "current_accelerator",
    "is_trainium",
    "is_nvidia",
    # Providers
    "AWS",
    "DigitalOcean",
    "Verda",
    # Provider selection
    "SelectionStrategy",
    "ProviderSelector",
    "SelectionLike",
    "ProviderLike",
    "select_first",
    "select_cheapest",
    "select_available",
    "NoAvailableProviderError",
    "AllProvidersFailedError",
    # Volumes
    "Volume",
    "S3Volume",
    # Cluster utilities
    "InstanceInfo",
    "instance_info",
    # Data sharding utilities
    "shard",
    "shard_iterator",
    "DistributedSampler",
    "integrations",
    "conc",
    # Version
    "__version__",
]


# =============================================================================
# Module class hack for `>> sky` syntax
# =============================================================================
#
# This enables the implicit pool execution syntax:
#
#     @sky.pool(provider=AWS(), ...)
#     def main():
#         result = train(data) >> sky  # Uses pool from context
#
# The magic: we replace the module's __class__ with a custom class that
# implements __rrshift__, so `pending >> sky` calls `sky.__rrshift__(pending)`.
#

import sys as _sys
from types import ModuleType as _ModuleType
from typing import Any as _Any


def _unpickle_skyward_module() -> _ModuleType:
    """Unpickle helper that returns the skyward module."""
    import skyward

    return skyward


class _SkywardModule(_ModuleType):
    """Custom module class that supports the >> and @ operators."""

    def __reduce__(self) -> tuple[object, tuple[()]]:
        """Make the module picklable by returning a function that imports it."""
        return (_unpickle_skyward_module, ())

    def __rrshift__(self, pending: _Any) -> _Any:
        """Execute pending computation on the current pool from context.

        This is called when using: `train(data) >> sky`

        The pool is retrieved from the context set by @sky.pool decorator.
        """
        from skyward._context import get_current_pool
        from skyward.pending import PendingBatch

        pool = get_current_pool()

        match pending:
            case PendingBatch():
                return pool.run_batch(pending)
            case _:
                return pool.run(pending)


# Replace this module's class with our custom class
_sys.modules[__name__].__class__ = _SkywardModule
