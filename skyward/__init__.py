"""Skyward - Execute Python functions on cloud compute.

Example:

    from skyward import compute, gather, ComputePool, AWS, Accelerator, Image

    @compute
    def train(data):
        return model.fit(data)

    pool = ComputePool(
        provider=AWS(),
        accelerator=Accelerator.NVIDIA.A100(),
        image=Image(pip=["torch"]),
    )

    with pool:
        result = train(data) >> pool
        r1, r2 = gather(train(d1), train(d2)) >> pool
"""

# Accelerator utilities
from skyward.accelerator import (
    is_nvidia,
    is_trainium,
)

# Distributed training decorators
from skyward import distributed

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

# Image
from skyward.image import Image

# Providers
from skyward.providers import AWS, DigitalOcean, Verda

# Types
from skyward.types import (
    GPU,
    NVIDIA,
    Accelerator,
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
    # Image
    "Image",
    # Distributed training
    "distributed",
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
    "select_instance",
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
    # Version
    "__version__",
]
