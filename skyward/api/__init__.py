"""Public API — self-contained type definitions and contracts.

All public types are importable from ``skyward.api``:

    from skyward.api import Spec, Options, Image, Pool, Session
"""

# ── Protocols & contracts ─────────────────────────────────────
from skyward.api.app import App as App
from skyward.api.compute import Compute as Compute

# ── Function & dispatch ──────────────────────────────────────
from skyward.api.context import sky as sky

# ── Distributed ──────────────────────────────────────────────
from skyward.api.distributed import Consistency as Consistency
from skyward.api.function import PendingFunction as PendingFunction
from skyward.api.function import PendingFunctionGroup as PendingFunctionGroup
from skyward.api.function import function as function
from skyward.api.function import gather as gather

# ── Logging ───────────────────────────────────────────────────
from skyward.api.logging import LogConfig as LogConfig
from skyward.api.logging import LogLevel as LogLevel

# ── Metrics ───────────────────────────────────────────────────
from skyward.api.metrics import Metric as Metric
from skyward.api.metrics import MetricsConfig as MetricsConfig

# ── Model types ───────────────────────────────────────────────
from skyward.api.model import Cluster as Cluster
from skyward.api.model import ClusterStatus as ClusterStatus
from skyward.api.model import Instance as Instance
from skyward.api.model import InstanceStatus as InstanceStatus
from skyward.api.model import InstanceType as InstanceType
from skyward.api.model import Offer as Offer

# ── Plugin system ─────────────────────────────────────────────
from skyward.api.plugin import Plugin as Plugin
from skyward.api.pool import Pool as Pool
from skyward.api.provider import ProviderConfig as ProviderConfig

# ── Runtime utilities ─────────────────────────────────────────
from skyward.api.runtime import CallbackWriter as CallbackWriter
from skyward.api.runtime import InstanceInfo as InstanceInfo
from skyward.api.runtime import instance_info as instance_info
from skyward.api.runtime import is_head as is_head
from skyward.api.runtime import redirect_output as redirect_output
from skyward.api.runtime import shard as shard
from skyward.api.runtime import silent as silent
from skyward.api.runtime import stderr as stderr
from skyward.api.runtime import stdout as stdout

# ── Stubs (behavioral types — real implementations in core/) ──
from skyward.api.session import Session as Session

# ── Spec types ────────────────────────────────────────────────
from skyward.api.spec import DEFAULT_IMAGE as DEFAULT_IMAGE
from skyward.api.spec import AllocationStrategy as AllocationStrategy
from skyward.api.spec import Architecture as Architecture
from skyward.api.spec import Image as Image
from skyward.api.spec import Options as Options
from skyward.api.spec import PipIndex as PipIndex
from skyward.api.spec import PoolSpec as PoolSpec
from skyward.api.spec import PoolState as PoolState
from skyward.api.spec import SelectionStrategy as SelectionStrategy
from skyward.api.spec import Spec as Spec
from skyward.api.spec import SpecKwargs as SpecKwargs
from skyward.api.spec import Volume as Volume
from skyward.api.spec import Worker as Worker
from skyward.api.spec import WorkerExecutor as WorkerExecutor

__all__ = [
    # Protocols
    "Pool",
    "ProviderConfig",
    # Spec
    "AllocationStrategy",
    "Architecture",
    "DEFAULT_IMAGE",
    "Image",
    "Options",
    "PipIndex",
    "PoolSpec",
    "PoolState",
    "SelectionStrategy",
    "Spec",
    "SpecKwargs",
    "Volume",
    "Worker",
    "WorkerExecutor",
    # Model
    "Cluster",
    "ClusterStatus",
    "Instance",
    "InstanceInfo",
    "InstanceStatus",
    "InstanceType",
    "Offer",
    # Function & dispatch
    "PendingFunction",
    "PendingFunctionGroup",
    "function",
    "gather",
    "sky",
    # Runtime
    "CallbackWriter",
    "instance_info",
    "is_head",
    "redirect_output",
    "shard",
    "silent",
    "stderr",
    "stdout",
    # Plugin
    "Plugin",
    # Metrics
    "Metric",
    "MetricsConfig",
    # Logging
    "LogConfig",
    "LogLevel",
    # Distributed
    "Consistency",
    # Stubs
    "Session",
    "Compute",
    "App",
]
