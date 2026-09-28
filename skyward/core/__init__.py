"""The client's surface: everything eager but ``Compute``, which is resolved on first use.

``Compute`` is the one name here that needs httpx, and a node has none: ``sky run``
executes a file on the node, and what its top level names — ``sky.app``, a
provider, an accelerator, ``sky.function`` — is imported there. The rest stays
eager because a lazy name that is also a submodule (``function``) would be
shadowed by that submodule the first time anything imported it.
"""

from typing import TYPE_CHECKING

from skyward.core.app import app
from skyward.core.containers import DockerImage
from skyward.core.context import sky
from skyward.core.errors import SkywardError, TaskFailedError, TaskIndeterminateError
from skyward.core.function import Group, Pending, Streaming, function, gather, stream
from skyward.core.provider import (
    AWS,
    GCP,
    Container,
    Hyperstack,
    JarvisLabs,
    Lambda,
    MassedCompute,
    Novita,
    Provider,
    RunPod,
    Salad,
    Scaleway,
    TensorDock,
    VastAI,
    Verda,
    Vultr,
)
from skyward.core.spec import Accelerator, Executor, HealthChecker, Nodes, Options, Port, Spec, Volume
from skyward.core.view import ComputeView, EventCallback, NodeView, PhaseView, TaskView
from skyward.shared.schemas import Image, MetricSpec, PipIndex

if TYPE_CHECKING:
    from skyward.core.compute import Compute

__all__ = [
    "AWS",
    "GCP",
    "Accelerator",
    "Compute",
    "ComputeView",
    "Container",
    "DockerImage",
    "EventCallback",
    "Executor",
    "Group",
    "HealthChecker",
    "Hyperstack",
    "Image",
    "JarvisLabs",
    "Lambda",
    "MassedCompute",
    "MetricSpec",
    "NodeView",
    "Nodes",
    "Novita",
    "Options",
    "Pending",
    "PhaseView",
    "PipIndex",
    "Port",
    "Provider",
    "RunPod",
    "Salad",
    "Scaleway",
    "SkywardError",
    "Spec",
    "Streaming",
    "TaskFailedError",
    "TaskIndeterminateError",
    "TaskView",
    "TensorDock",
    "VastAI",
    "Verda",
    "Volume",
    "Vultr",
    "app",
    "function",
    "gather",
    "sky",
    "stream",
]


def __getattr__(name: str) -> object:
    match name:
        case "Compute":
            from skyward.core.compute import Compute

            globals()[name] = Compute
            return Compute
        case _:
            raise AttributeError(f"module 'skyward.core' has no attribute '{name}'")
