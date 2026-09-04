"""Skyward — write a function, run it on someone else's machines.

    import skyward as sky

    @sky.function
    def train(data):
        return model.fit(data)

    with sky.Compute(provider=sky.AWS(), accelerator=sky.accelerators.A100(), nodes=4) as pool:
        model = train(data) >> pool

Nothing is imported until it is asked for. The package is also what a *node*
imports to run the user's code, and a node has no web framework, no HTTP client
and no business acquiring one: reaching ``sky.Compute`` pulls the client in,
reaching nothing pulls nothing.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skyward.core import (
        AWS,
        GCP,
        Accelerator,
        Compute,
        ComputeView,
        Container,
        DockerImage,
        EventCallback,
        Executor,
        Group,
        HealthChecker,
        Hyperstack,
        Image,
        JarvisLabs,
        Lambda,
        MassedCompute,
        MetricSpec,
        Nodes,
        NodeView,
        Novita,
        Options,
        Pending,
        PhaseView,
        PipIndex,
        Port,
        Provider,
        RunPod,
        Salad,
        Scaleway,
        SkywardError,
        Spec,
        Streaming,
        TaskFailedError,
        TaskIndeterminateError,
        TaskView,
        TensorDock,
        VastAI,
        Verda,
        Volume,
        Vultr,
        accelerators,
        function,
        gather,
        sky,
        stream,
    )
    from skyward.shared import observability, time
    from skyward.shared.events import (
        ComputeAbandoned,
        ComputeAdopted,
        ComputeBound,
        ComputeCreated,
        ComputeDegraded,
        ComputeDeleted,
        ComputeDeleting,
        ComputeDeletionFailed,
        ComputeProvisioning,
        ComputeReady,
        ConsoleEvent,
        CostEvent,
        Event,
        GenerationApplied,
        GenerationCreated,
        LeaseClaimed,
        LeaseReleased,
        MetricEvent,
        NodeEvent,
        PhaseEvent,
        ProgressEvent,
        StraysTerminated,
        TaskEvent,
    )
    from skyward.shared.schemas import ComputeState, NodeState, TaskState
    from skyward.worker import metrics, plugins, storage
    from skyward.worker.api import CallbackWriter, Info, instance_info, is_head, redirect_output, shard, silent, stderr, stdout
    from skyward.worker.distributed import Consistency, DistributedRegistry, barrier, counter, dict, lock, queue, registry, set
    from skyward.worker.storage import Storage

RUNTIME = ("CallbackWriter", "Info", "instance_info", "is_head", "redirect_output", "shard", "silent", "stderr", "stdout")
"""What a function asks while it is running, and the only names a node ever reaches for.

They resolve out of ``skyward.worker.api``, which imports the standard library and
nothing else. Routing them through the SDK would put httpx on a machine whose only
job is to run somebody's training loop.
"""

SHARED = ("Consistency", "DistributedRegistry", "barrier", "counter", "dict", "lock", "queue", "registry", "set")
"""The compute's own state, out of ``skyward.worker.distributed``. Same reason."""

EVENTS = (
    "ComputeAbandoned",
    "ComputeAdopted",
    "ComputeBound",
    "ComputeCreated",
    "ComputeDegraded",
    "ComputeDeleted",
    "ComputeDeleting",
    "ComputeDeletionFailed",
    "ComputeProvisioning",
    "ComputeReady",
    "ConsoleEvent",
    "CostEvent",
    "Event",
    "GenerationApplied",
    "GenerationCreated",
    "LeaseClaimed",
    "LeaseReleased",
    "MetricEvent",
    "NodeEvent",
    "PhaseEvent",
    "ProgressEvent",
    "StraysTerminated",
    "TaskEvent",
)
"""The event stream's vocabulary, out of ``skyward.shared.events``.

What a ``callbacks=`` handler matches on. Resolved from ``shared`` rather than
``core`` because the events are wire types — msgspec and the standard library,
nothing a bare node could not import.
"""

SCHEMAS = ("ComputeState", "NodeState", "TaskState")
"""The state words, out of ``skyward.shared.schemas``, for the same reason."""

__all__ = [
    "AWS",
    "Accelerator",
    "CallbackWriter",
    "Compute",
    "ComputeAbandoned",
    "ComputeAdopted",
    "ComputeBound",
    "ComputeCreated",
    "ComputeDegraded",
    "ComputeDeleted",
    "ComputeDeleting",
    "ComputeDeletionFailed",
    "ComputeProvisioning",
    "ComputeReady",
    "ComputeState",
    "ComputeView",
    "Consistency",
    "ConsoleEvent",
    "Container",
    "CostEvent",
    "DistributedRegistry",
    "DockerImage",
    "Event",
    "EventCallback",
    "Executor",
    "GCP",
    "GenerationApplied",
    "GenerationCreated",
    "Group",
    "HealthChecker",
    "Hyperstack",
    "Image",
    "Info",
    "JarvisLabs",
    "Lambda",
    "LeaseClaimed",
    "LeaseReleased",
    "MassedCompute",
    "MetricEvent",
    "MetricSpec",
    "NodeEvent",
    "NodeState",
    "NodeView",
    "Nodes",
    "Novita",
    "Options",
    "Pending",
    "PhaseEvent",
    "PhaseView",
    "PipIndex",
    "Port",
    "ProgressEvent",
    "Provider",
    "RunPod",
    "Salad",
    "Scaleway",
    "SkywardError",
    "Spec",
    "Storage",
    "StraysTerminated",
    "Streaming",
    "TaskEvent",
    "TaskFailedError",
    "TaskIndeterminateError",
    "TaskState",
    "TaskView",
    "TensorDock",
    "VastAI",
    "Verda",
    "Volume",
    "Vultr",
    "accelerators",
    "barrier",
    "counter",
    "dict",
    "function",
    "gather",
    "instance_info",
    "is_head",
    "lock",
    "metrics",
    "observability",
    "plugins",
    "queue",
    "redirect_output",
    "registry",
    "set",
    "shard",
    "silent",
    "sky",
    "stderr",
    "stdout",
    "storage",
    "stream",
    "time",
]


def __getattr__(name: str) -> object:
    match name:
        case "accelerators":
            value = importlib.import_module("skyward.core.accelerators")
        case "observability" | "time":
            value = importlib.import_module(f"skyward.shared.{name}")
        case "metrics" | "plugins" | "storage":
            value = importlib.import_module(f"skyward.worker.{name}")
        case "Storage":
            value = importlib.import_module("skyward.worker.storage").Storage
        case _ if name in EVENTS:
            value = getattr(importlib.import_module("skyward.shared.events"), name)
        case _ if name in SCHEMAS:
            value = getattr(importlib.import_module("skyward.shared.schemas"), name)
        case _ if name in RUNTIME:
            value = getattr(importlib.import_module("skyward.worker.api"), name)
        case _ if name in SHARED:
            value = getattr(importlib.import_module("skyward.worker.distributed"), name)
        case _ if name in __all__:
            try:
                value = getattr(importlib.import_module("skyward.core"), name)
            except ModuleNotFoundError as error:
                raise ImportError(f"'{name}' is part of the client — install 'skyward[client]'") from error
        case _:
            raise AttributeError(f"module 'skyward' has no attribute '{name}'")

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
