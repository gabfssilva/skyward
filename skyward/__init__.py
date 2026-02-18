"""Skyward — distributed compute orchestration for ML/AI.

    import skyward as sky

    @sky.compute
    def train(data):
        return model.fit(data)

    with sky.ComputePool(provider=sky.AWS(), accelerator="A100") as pool:
        result = train(data) >> pool
"""

try:
    from skyward._version import __version__ as __version__
except ModuleNotFoundError:
    __version__: str = "0.0.0+unknown"

from skyward import accelerators as accelerators
from skyward import integrations as integrations
from skyward.observability import LogConfig
from skyward.observability import metrics as metrics

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
    InstanceMetadata,
    InstancePreempted,
    InstanceProvisioned,
    InstanceReplaced,
    InstanceRequested,
    Log,
    Metric,
    NodeId,
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
    ComputePool,
    Image,
    InflightStrategy,
    InstanceInfo,
    PendingCompute,
    PendingComputeGroup,
    PoolSpec,
    compute,
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
from .app import App
from .distributed import (
    barrier,
    counter,
    dict,
    lock,
    queue,
    set,
)
from .providers import AWS, Container, RunPod, VastAI, Verda

__all__ = [
    "__version__",
    "App",
    "sky",
    "pool",
    "compute",
    "gather",
    "ComputePool",
    "PendingCompute",
    "PendingComputeGroup",
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
    "RunPod",
    "VastAI",
    "Verda",
    "Image",
    "DEFAULT_IMAGE",
    "PoolSpec",
    "AllocationStrategy",
    "InflightStrategy",
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
    "InstanceMetadata",
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
    "integrations",
]
