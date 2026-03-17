"""Skyward — distributed compute orchestration for ML/AI.

    import skyward as sky

    @sky.function
    def train(data):
        return model.fit(data)

    with sky.Compute(provider=sky.AWS(), accelerator="A100") as compute:
        result = train(data) >> compute
"""

from __future__ import annotations

from typing import Any

# ── Sub-module namespaces ─────────────────────────────────────
from skyward import accelerators as accelerators
from skyward import plugins as plugins
from skyward import storage as storage

# ── Re-exported events (frozen dataclasses — not in api/) ─────
from skyward.actors.messages import ClusterDestroyed as ClusterDestroyed
from skyward.actors.messages import ClusterId as ClusterId
from skyward.actors.messages import ClusterProvisioned as ClusterProvisioned
from skyward.actors.messages import ClusterReady as ClusterReady
from skyward.actors.messages import ClusterRequested as ClusterRequested
from skyward.actors.messages import Error as Error
from skyward.actors.messages import Event as Event
from skyward.actors.messages import Fact as Fact
from skyward.actors.messages import InstanceBootstrapped as InstanceBootstrapped
from skyward.actors.messages import InstanceDestroyed as InstanceDestroyed
from skyward.actors.messages import InstanceId as InstanceId
from skyward.actors.messages import InstancePreempted as InstancePreempted
from skyward.actors.messages import InstanceProvisioned as InstanceProvisioned
from skyward.actors.messages import InstanceReplaced as InstanceReplaced
from skyward.actors.messages import InstanceRequested as InstanceRequested
from skyward.actors.messages import Log as Log
from skyward.actors.messages import Metric as Metric
from skyward.actors.messages import NodeId as NodeId
from skyward.actors.messages import NodeInstance as NodeInstance
from skyward.actors.messages import ProviderName as ProviderName
from skyward.actors.messages import Request as Request
from skyward.actors.messages import RequestId as RequestId
from skyward.actors.messages import ShutdownCompleted as ShutdownCompleted
from skyward.actors.messages import ShutdownRequested as ShutdownRequested
from skyward.actors.messages import TaskCompleted as TaskCompleted
from skyward.actors.messages import TaskStarted as TaskStarted

# ── Re-exported from skyward.api ─────────────────────────────
from skyward.api.app import App as App
from skyward.api.compute import Compute as Compute
from skyward.api.context import sky as sky
from skyward.api.distributed import Consistency as Consistency
from skyward.api.function import PendingFunction as PendingFunction
from skyward.api.function import PendingFunctionGroup as PendingFunctionGroup
from skyward.api.function import function as function
from skyward.api.function import gather as gather
from skyward.api.logging import LogConfig as LogConfig
from skyward.api.model import Cluster as Cluster
from skyward.api.model import Instance as Instance
from skyward.api.model import InstanceType as InstanceType
from skyward.api.model import Offer as Offer
from skyward.api.pool import Pool as Pool
from skyward.api.provider import ProviderConfig as ProviderConfig
from skyward.api.runtime import CallbackWriter as CallbackWriter
from skyward.api.runtime import InstanceInfo as InstanceInfo
from skyward.api.runtime import instance_info as instance_info
from skyward.api.runtime import is_head as is_head
from skyward.api.runtime import redirect_output as redirect_output
from skyward.api.runtime import shard as shard
from skyward.api.runtime import silent as silent
from skyward.api.runtime import stderr as stderr
from skyward.api.runtime import stdout as stdout
from skyward.api.session import Session as Session
from skyward.api.spec import DEFAULT_IMAGE as DEFAULT_IMAGE
from skyward.api.spec import AllocationStrategy as AllocationStrategy
from skyward.api.spec import Image as Image
from skyward.api.spec import Nodes as Nodes
from skyward.api.spec import NodeSpec as NodeSpec
from skyward.api.spec import Options as Options
from skyward.api.spec import PipIndex as PipIndex
from skyward.api.spec import PoolSpec as PoolSpec
from skyward.api.spec import SelectionStrategy as SelectionStrategy
from skyward.api.spec import Spec as Spec
from skyward.api.spec import SpecKwargs as SpecKwargs
from skyward.api.spec import Volume as Volume
from skyward.api.spec import Worker as Worker
from skyward.api.spec import WorkerExecutor as WorkerExecutor

# ── Re-exported distributed proxies & factories ──────────────
from skyward.distributed import barrier as barrier
from skyward.distributed import counter as counter
from skyward.distributed import dict as dict
from skyward.distributed import lock as lock
from skyward.distributed import queue as queue
from skyward.distributed import set as set

# ── Observability ────────────────────────────────────────────
from skyward.observability import metrics as metrics

# ── Offers ───────────────────────────────────────────────────
from skyward.offers.repository import OfferRepository as OfferRepository

# ── Re-exported providers ────────────────────────────────────
from skyward.providers import AWS as AWS
from skyward.providers import GCP as GCP
from skyward.providers import Container as Container
from skyward.providers import Hyperstack as Hyperstack
from skyward.providers import JarvisLabs as JarvisLabs
from skyward.providers import RunPod as RunPod
from skyward.providers import TensorDock as TensorDock
from skyward.providers import VastAI as VastAI
from skyward.providers import Verda as Verda
from skyward.providers import Vultr as Vultr

# ── Storage ──────────────────────────────────────────────────
from skyward.storage import Storage as Storage

# ── Version ──────────────────────────────────────────────────

__version__: str

# ── Offers function ──────────────────────────────────────────

async def offers(providers: list[Any]) -> OfferRepository:
    """Load the GPU offer catalog into a queryable repository."""
    ...

# ── __all__ ──────────────────────────────────────────────────

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
    "Options",
    "PipIndex",
    "DEFAULT_IMAGE",
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
    "Nodes",
    "NodeSpec",
    "plugins",
]
