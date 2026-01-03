"""Skyward - Execute Python functions on cloud compute.

Example:

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
"""

# Accelerator utilities
import skyward.conc as conc
import skyward.integrations as integrations
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

# Allocation strategies
from skyward.spec import Allocation, AllocationLike

# Providers
from skyward.providers import AWS, DigitalOcean, Verda

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
    # Image
    "Image",
    # Callback system
    "emit",
    "use_callback",
    "compose",
    "Callback",
    # === Types ===
    "Instance",
    "InstanceSpec",
    "ExitedInstance",
    "Provider",
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
