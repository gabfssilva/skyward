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
from skyward import containers as containers
from skyward import plugins as plugins
from skyward import storage as storage
from skyward import time as time

# ── Re-exported from skyward.api ─────────────────────────────
from skyward.api.app import App as App
from skyward.api.application import Application as Application
from skyward.api.application import app as app
from skyward.api.compute import Compute as Compute
from skyward.api.context import sky as sky
from skyward.api.distributed import Consistency as Consistency

# ── Re-exported events (frozen dataclasses — not in api/) ─────
from skyward.api.facts import ClusterDestroyed as ClusterDestroyed
from skyward.api.facts import ClusterId as ClusterId
from skyward.api.facts import ClusterProvisioned as ClusterProvisioned
from skyward.api.facts import ClusterReady as ClusterReady
from skyward.api.facts import Error as Error
from skyward.api.facts import Event as Event
from skyward.api.facts import Fact as Fact
from skyward.api.facts import InstanceBootstrapped as InstanceBootstrapped
from skyward.api.facts import InstanceDestroyed as InstanceDestroyed
from skyward.api.facts import InstanceId as InstanceId
from skyward.api.facts import InstancePreempted as InstancePreempted
from skyward.api.facts import InstanceProvisioned as InstanceProvisioned
from skyward.api.facts import InstanceReplaced as InstanceReplaced
from skyward.api.facts import Log as Log
from skyward.api.facts import Metric as Metric
from skyward.api.facts import NodeId as NodeId
from skyward.api.facts import NodeInstance as NodeInstance
from skyward.api.facts import ProviderName as ProviderName
from skyward.api.facts import RequestId as RequestId
from skyward.api.facts import TaskCompleted as TaskCompleted
from skyward.api.facts import TaskStarted as TaskStarted
from skyward.api.function import PendingFunction as PendingFunction
from skyward.api.function import PendingFunctionGroup as PendingFunctionGroup
from skyward.api.function import function as function
from skyward.api.function import gather as gather
from skyward.api.health import HealthChecker as HealthChecker
from skyward.api.logging import LogConfig as LogConfig
from skyward.api.main import main as main
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
from skyward.api.spec import Port as Port
from skyward.api.spec import SelectionStrategy as SelectionStrategy
from skyward.api.spec import Spec as Spec
from skyward.api.spec import SpecKwargs as SpecKwargs
from skyward.api.spec import Volume as Volume
from skyward.api.spec import Worker as Worker
from skyward.api.spec import WorkerExecutor as WorkerExecutor
from skyward.containers import DockerImage as DockerImage

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
from skyward.providers import LambdaCloud as LambdaCloud
from skyward.providers import MassedCompute as MassedCompute
from skyward.providers import Novita as Novita
from skyward.providers import RunPod as RunPod
from skyward.providers import Scaleway as Scaleway
from skyward.providers import TensorDock as TensorDock
from skyward.providers import VastAI as VastAI
from skyward.providers import Verda as Verda
from skyward.providers import Vultr as Vultr
from skyward.server.client import Client as Client

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
    "Application",
    "Client",
    "Compute",
    "Session",
    "sky",
    "app",
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
    "LambdaCloud",
    "MassedCompute",
    "Novita",
    "RunPod",
    "Scaleway",
    "TensorDock",
    "VastAI",
    "Verda",
    "Vultr",
    "HealthChecker",
    "Image",
    "Options",
    "PipIndex",
    "DEFAULT_IMAGE",
    "PoolSpec",
    "Port",
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
    "Fact",
    "Event",
    "metrics",
    "LogConfig",
    "accelerators",
    "offers",
    "OfferRepository",
    "Nodes",
    "NodeSpec",
    "main",
    "plugins",
    "containers",
    "DockerImage",
    "time",
]
