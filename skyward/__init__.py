"""Skyward - Execute Python functions on cloud compute.

Example:

    from skyward import compute, gather, ComputePool, AWS

    @compute
    def train(data):
        return model.fit(data)

    pool = ComputePool(provider=AWS(), accelerator="A100", pip=["torch"])

    with pool:
        result = train(data) >> pool
        r1, r2 = gather(train(d1), train(d2)) >> pool
"""

import logging

# Accelerator utilities
from skyward.accelerator import is_nvidia, is_trainium

# Event Bus
from skyward.bus import EventBus

# Cluster utilities
from skyward.cluster import InstanceInfo, instance_info

# Data sharding utilities
from skyward.data import DistributedSampler, shard, shard_iterator

# Events (ADT)
from skyward.events import (
    BootstrapCompleted,
    BootstrapProgress,
    BootstrapStarting,
    Error,
    EventCallback,
    InfraCreated,
    InfraCreating,
    InstanceLaunching,
    InstanceProvisioned,
    InstanceStopping,
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

# Providers
from skyward.providers.aws import AWS
from skyward.providers.digitalocean import DigitalOcean

# Types
from skyward.types import (
    GPU,
    NVIDIA,
    Accelerator,
    ComputeSpec,
    ExitedInstance,
    Instance,
    Provider,
    Trainium,
    current_accelerator,
)

# Volumes
from skyward.volume import S3Volume, Volume

__version__ = "0.2.0"


def set_log_level(level: str = "INFO") -> None:
    """Set the logging level for Skyward."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger("skyward")
    logger.setLevel(numeric_level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)


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
    # Event Bus
    "EventBus",
    # === Types ===
    "Instance",
    "ExitedInstance",
    "Provider",
    "ComputeSpec",
    # Events (ADT)
    "SkywardEvent",
    "EventCallback",
    "InfraCreating",
    "InfraCreated",
    "InstanceLaunching",
    "InstanceProvisioned",
    "BootstrapStarting",
    "BootstrapProgress",
    "BootstrapCompleted",
    "Metrics",
    "InstanceStopping",
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
    # Configuration
    "set_log_level",
    "__version__",
]
