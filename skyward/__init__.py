"""Skyward v2 - Event-driven compute orchestration.

Usage as module (recommended):

    import skyward as sky

    @sky.compute
    def train(data):
        return model.fit(data)

    @sky.pool(provider=sky.AWS(), accelerator="A100", nodes=4)
    def main():
        result = train(data) >> sky          # execute on one node
        results = train(data) @ sky          # broadcast to all nodes
        a, b = (task1() & task2()) >> sky    # parallel execution

Or as context manager:

    import skyward as sky

    @sky.compute
    def train(data): ...

    with sky.pool(provider=sky.AWS(), accelerator="A100") as p:
        result = train(data) >> sky  # or >> p
"""

from __future__ import annotations

# =============================================================================
# User-facing API (sync facade) - PRIMARY EXPORTS
# =============================================================================

from .facade import (
    # The sky singleton for >> sky / @ sky
    sky,
    # Core functions
    pool,
    compute,
    gather,
    # Types
    SyncComputePool,
    PendingCompute,
    PendingComputeGroup,
    # Utilities
    InstanceInfo,
    instance_info,
    shard,
)

# =============================================================================
# Providers (config classes only - no SDK dependencies)
# =============================================================================

from .providers import AWS, VastAI, Verda

# NOTE: Handlers and modules are NOT imported here to avoid SDK deps.
# Import them explicitly when needed:
#   from skyward.providers.aws import AWSHandler, AWSModule

# =============================================================================
# Image configuration
# =============================================================================

from .image import Image, DEFAULT_IMAGE

# =============================================================================
# Metrics (lazy-loaded submodule)
# =============================================================================

# Use: sky.metrics.CPU(), sky.metrics.GPU(), sky.metrics.Default()
# Lazy to avoid importing metric specs until needed
from skyward import metrics as metrics

# =============================================================================
# Events - the language of the system
# =============================================================================

from .events import (
    # Type aliases
    ClusterId,
    InstanceId,
    NodeId,
    ProviderName,
    RequestId,
    # Value objects
    InstanceMetadata,
    # Requests
    ClusterRequested,
    InstanceRequested,
    ShutdownRequested,
    # Facts
    ClusterDestroyed,
    ClusterProvisioned,
    ClusterReady,
    Error,
    InstanceBootstrapped,
    InstanceDestroyed,
    InstancePreempted,
    InstanceProvisioned,
    InstanceReplaced,
    Log,
    Metric,
    NodeReady,
    TaskCompleted,
    TaskStarted,
    # Unions
    Event,
    Fact,
    Request,
)

# =============================================================================
# Bus - async event routing
# =============================================================================

from .bus import AsyncEventBus

# =============================================================================
# App - decorators and bootstrap
# =============================================================================

from .app import (
    app_context,
    clear_registries,
    component,
    create_app,
    monitor,
    MonitorManager,
    on,
)

# =============================================================================
# Audit - observability decorator
# =============================================================================

from .audit import audit

# =============================================================================
# Specs - configuration
# =============================================================================

from .spec import (
    AllocationStrategy,
    PoolSpec,
)

# =============================================================================
# Accelerators - GPU/TPU specifications
# =============================================================================

from .accelerators import Accelerator

# Lazy-loaded submodule: sky.accelerators.H100(), sky.accelerators.T4(), etc.
from skyward import accelerators as accelerators

# =============================================================================
# Protocols - interfaces
# =============================================================================

from .protocols import (
    Executor,
    HealthChecker,
    PreemptionChecker,
    Serializable,
    Transport,
    TransportFactory,
)

# =============================================================================
# Components
# =============================================================================

from .node import Node, NodeState
from .pool import ComputePool, PoolState

# =============================================================================
# Monitors
# =============================================================================

from .monitors import InstanceRegistry, MonitorModule

# =============================================================================
# Transport
# =============================================================================

from .transport import SSHTransport

# =============================================================================
# Executor
# =============================================================================

from .executor import Executor

# =============================================================================
# Retry
# =============================================================================

from .retry import (
    all_of,
    any_of,
    on_exception_message,
    on_status_code,
    retry,
)

# =============================================================================
# Throttle
# =============================================================================

from .throttle import (
    Limiter,
    ThrottleError,
    throttle,
)

# =============================================================================
# Output Control
# =============================================================================

from .stdout import (
    stdout,
    stderr,
    silent,
    is_head,
    CallbackWriter,
    redirect_output,
)

# =============================================================================
# Integrations (lazy-loaded submodule)
# =============================================================================

# Use: from skyward import integrations
#      from skyward.integrations import keras
# Lazy to avoid requiring torch/jax/tensorflow at import time
from skyward import integrations as integrations

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # =================================================================
    # Primary User API (import skyward as sky)
    # =================================================================
    "sky",           # singleton for >> sky / @ sky
    "pool",          # decorator or context manager
    "compute",       # @compute decorator
    "gather",        # gather() for parallel execution
    "InstanceInfo",  # runtime info type
    "instance_info", # get instance info inside @compute
    "shard",         # shard data across nodes
    # Types
    "SyncComputePool",
    "PendingCompute",
    "PendingComputeGroup",
    # =================================================================
    # Providers (config classes only - import handlers explicitly)
    # =================================================================
    "AWS",
    "VastAI",
    "Verda",
    # =================================================================
    # Image
    # =================================================================
    "Image",
    "DEFAULT_IMAGE",
    # =================================================================
    # Metrics (submodule)
    # =================================================================
    "metrics",  # submodule: sky.metrics.CPU(), sky.metrics.GPU(), etc.
    # =================================================================
    # Events - Type aliases
    # =================================================================
    "RequestId",
    "ClusterId",
    "InstanceId",
    "NodeId",
    "ProviderName",
    # Events - Value objects
    "InstanceMetadata",
    # Events - Requests
    "ClusterRequested",
    "InstanceRequested",
    "ShutdownRequested",
    # Events - Facts
    "ClusterProvisioned",
    "InstanceProvisioned",
    "InstanceBootstrapped",
    "InstancePreempted",
    "InstanceReplaced",
    "InstanceDestroyed",
    "NodeReady",
    "ClusterReady",
    "ClusterDestroyed",
    "TaskStarted",
    "TaskCompleted",
    "Metric",
    "Log",
    "Error",
    # Events - Unions
    "Request",
    "Fact",
    "Event",
    # =================================================================
    # Bus
    # =================================================================
    "AsyncEventBus",
    # =================================================================
    # App
    # =================================================================
    "component",
    "on",
    "monitor",
    "create_app",
    "app_context",
    "MonitorManager",
    "clear_registries",
    # =================================================================
    # Audit
    # =================================================================
    "audit",
    # =================================================================
    # Specs
    # =================================================================
    "PoolSpec",
    "AllocationStrategy",
    # =================================================================
    # Accelerators
    # =================================================================
    "Accelerator",
    "accelerators",  # submodule: sky.accelerators.H100(), etc.
    # =================================================================
    # Protocols
    # =================================================================
    "Transport",
    "Executor",
    "TransportFactory",
    "HealthChecker",
    "PreemptionChecker",
    "Serializable",
    # =================================================================
    # Components
    # =================================================================
    "Node",
    "NodeState",
    "ComputePool",
    "PoolState",
    # =================================================================
    # Monitors
    # =================================================================
    "InstanceRegistry",
    "MonitorModule",
    # =================================================================
    # Transport
    # =================================================================
    "SSHTransport",
    # =================================================================
    # Retry
    # =================================================================
    "retry",
    "on_status_code",
    "on_exception_message",
    "any_of",
    "all_of",
    # =================================================================
    # Throttle
    # =================================================================
    "throttle",
    "Limiter",
    "ThrottleError",
    # =================================================================
    # Output Control
    # =================================================================
    "stdout",
    "stderr",
    "silent",
    "is_head",
    "CallbackWriter",
    "redirect_output",
    # =================================================================
    # Integrations (submodule)
    # =================================================================
    "integrations",
]
