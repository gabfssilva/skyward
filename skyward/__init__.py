"""Skyward — distributed compute orchestration for ML/AI.

    import skyward as sky

    @sky.function
    def train(data):
        return model.fit(data)

    with sky.ComputePool(provider=sky.AWS(), accelerator=sky.accelerators.A100()) as pool:
        result = train(data) >> pool
"""
from typing import Any

try:
    from skyward._version import __version__ as __version__
except ModuleNotFoundError:
    __version__: str = "0.0.0+unknown"

from skyward import accelerators as accelerators
from skyward import plugins as plugins
from skyward import storage as storage
from skyward.observability import LogConfig
from skyward.observability import metrics as metrics
from skyward.storage import Storage as Storage

from .actors.messages import (
    ClusterDestroyed,
    ClusterId,
    ClusterProvisioned,
    ClusterReady,
    ClusterRequested,
    Error,
    Event,
    Fact,
    InstanceBootstrapped,
    InstanceDestroyed,
    InstanceId,
    InstancePreempted,
    InstanceProvisioned,
    InstanceReplaced,
    InstanceRequested,
    Log,
    Metric,
    NodeId,
    NodeInstance,
    ProviderName,
    Request,
    RequestId,
    ShutdownCompleted,
    ShutdownRequested,
    TaskCompleted,
    TaskStarted,
)
from .api import (
    DEFAULT_IMAGE,
    AllocationStrategy,
    CallbackWriter,
    Compute,
    ComputePool,
    Image,
    InstanceInfo,
    PendingFunction,
    PendingFunctionGroup,
    PipIndex,
    PoolSpec,
    Session,
    function,
    gather,
    instance_info,
    is_head,
    pool,
    redirect_output,
    shard,
    silent,
    sky,
    stderr,
    stdout,
)
from .api import InstanceType as InstanceType
from .api import Offer as Offer
from .api import SelectionStrategy as SelectionStrategy
from .api import Spec as Spec
from .api import Volume as Volume
from .api import Worker as Worker
from .api import WorkerExecutor as WorkerExecutor
from .app import App
from .distributed import (
    barrier,
    counter,
    dict,
    lock,
    queue,
    set,
)
from .offers.repository import OfferRepository
from .providers import AWS, GCP, Container, Hyperstack, RunPod, TensorDock, VastAI, Verda


async def offers(providers: list[Any]) -> OfferRepository:
    """Load the GPU offer catalog into a queryable repository.

    Usage::

        import skyward as sky

        repo = await sky.offers()
        offer = repo.accelerator("A100").spot().cheapest()
    """
    return await OfferRepository.create(providers)

__all__ = [
    "__version__",
    "App",
    "Compute",
    "Session",
    "sky",
    "pool",
    "function",
    "gather",
    "ComputePool",
    "PendingFunction",
    "PendingFunctionGroup",
    "InstanceInfo",
    "instance_info",
    "shard",
    "stdout",
    "stderr",
    "silent",
    "is_head",
    "CallbackWriter",
    "redirect_output",
    "AWS",
    "Container",
    "GCP",
    "Hyperstack",
    "RunPod",
    "TensorDock",
    "VastAI",
    "Verda",
    "Image",
    "PipIndex",
    "DEFAULT_IMAGE",
    "PoolSpec",
    "AllocationStrategy",
    "InstanceType",
    "Offer",
    "SelectionStrategy",
    "Spec",
    "Volume",
    "Storage",
    "storage",
    "Worker",
    "WorkerExecutor",
    "dict",
    "set",
    "counter",
    "queue",
    "barrier",
    "lock",
    "RequestId",
    "ClusterId",
    "InstanceId",
    "NodeId",
    "ProviderName",
    "NodeInstance",
    "ClusterRequested",
    "InstanceRequested",
    "ShutdownCompleted",
    "ShutdownRequested",
    "ClusterProvisioned",
    "InstanceProvisioned",
    "InstanceBootstrapped",
    "InstancePreempted",
    "InstanceReplaced",
    "InstanceDestroyed",
    "ClusterReady",
    "ClusterDestroyed",
    "TaskStarted",
    "TaskCompleted",
    "Metric",
    "Log",
    "Error",
    "Request",
    "Fact",
    "Event",
    "metrics",
    "LogConfig",
    "accelerators",
    "offers",
    "OfferRepository",
    "plugins",
]
