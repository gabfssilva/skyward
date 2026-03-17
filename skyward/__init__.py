"""Skyward — distributed compute orchestration for ML/AI.

    import skyward as sky

    @sky.function
    def train(data):
        return model.fit(data)

    with sky.Compute(provider=sky.AWS(), accelerator="A100") as compute:
        result = train(data) >> compute
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
from .api.pool import Pool as Pool
from .app import App
from .core import (
    DEFAULT_IMAGE,
    AllocationStrategy,
    CallbackWriter,
    Compute,
    Image,
    InstanceInfo,
    Options,
    PendingFunction,
    PendingFunctionGroup,
    PipIndex,
    PoolSpec,
    Session,
    function,
    gather,
    instance_info,
    is_head,
    redirect_output,
    shard,
    silent,
    sky,
    stderr,
    stdout,
)
from .core import InstanceType as InstanceType
from .core import Nodes as Nodes
from .core import Offer as Offer
from .core import SelectionStrategy as SelectionStrategy
from .core import Spec as Spec
from .core import SpecKwargs as SpecKwargs
from .core import Volume as Volume
from .core import Worker as Worker
from .core import WorkerExecutor as WorkerExecutor
from .distributed import (
    barrier,
    counter,
    dict,
    lock,
    queue,
    set,
)
from .offers.repository import OfferRepository
from .providers import AWS, GCP, Container, Hyperstack, JarvisLabs, RunPod, TensorDock, VastAI, Verda, Vultr


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
    "function",
    "gather",
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
    "JarvisLabs",
    "RunPod",
    "TensorDock",
    "VastAI",
    "Verda",
    "Vultr",
    "Image",
    "Nodes",
    "Options",
    "PipIndex",
    "DEFAULT_IMAGE",
    "Pool",
    "PoolSpec",
    "AllocationStrategy",
    "InstanceType",
    "Offer",
    "SelectionStrategy",
    "Spec",
    "SpecKwargs",
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
