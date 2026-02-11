"""Skyward — distributed compute orchestration for ML/AI.

    import skyward as sky

    @sky.compute
    def train(data):
        return model.fit(data)

    with sky.pool(provider=sky.AWS(), accelerator="A100") as p:
        result = train(data) >> sky
"""

from .api import (
    AllocationStrategy,
    CallbackWriter,
    DEFAULT_IMAGE,
    Image,
    InstanceInfo,
    PendingCompute,
    PendingComputeGroup,
    PoolSpec,
    SyncComputePool,
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

from .providers import AWS, RunPod, VastAI, Verda

from .distributed import (
    barrier,
    counter,
    dict,
    lock,
    queue,
    set,
)

from .actors.messages import (
    ClusterDestroyed,
    ClusterProvisioned,
    ClusterReady,
    ClusterRequested,
    ClusterId,
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
    ShutdownRequested,
    TaskCompleted,
    TaskStarted,
)

from skyward.observability import metrics as metrics
from skyward import accelerators as accelerators
from skyward import integrations as integrations

__all__ = [
    "sky",
    "pool",
    "compute",
    "gather",
    "SyncComputePool",
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
    "RunPod",
    "VastAI",
    "Verda",
    "Image",
    "DEFAULT_IMAGE",
    "PoolSpec",
    "AllocationStrategy",
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
    "accelerators",
    "integrations",
]
