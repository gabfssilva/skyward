"""Skyward — write a function, run it on someone else's machines.

    import skyward2 as sky

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
    from skyward2 import accelerators
    from skyward2.runtime.api import Info, instance_info, is_head, shard, silent, stderr, stdout
    from skyward2.sdk import (
        AWS,
        GCP,
        Accelerator,
        Compute,
        Container,
        Group,
        Hyperstack,
        Image,
        JarvisLabs,
        Lambda,
        MassedCompute,
        Nodes,
        Novita,
        Pending,
        Provider,
        RunPod,
        Scaleway,
        SkywardError,
        Spec,
        TaskFailedError,
        TaskIndeterminateError,
        TensorDock,
        VastAI,
        Verda,
        Vultr,
        function,
        gather,
    )

RUNTIME = ("Info", "instance_info", "is_head", "shard", "silent", "stderr", "stdout")
"""What a function asks while it is running, and the only names a node ever reaches for.

They resolve out of ``skyward2.runtime.api``, which imports the standard library and
nothing else. Routing them through the SDK would put httpx on a machine whose only
job is to run somebody's training loop.
"""

__all__ = [
    "AWS",
    "GCP",
    "Accelerator",
    "Compute",
    "Container",
    "Group",
    "Hyperstack",
    "Image",
    "Info",
    "JarvisLabs",
    "Lambda",
    "MassedCompute",
    "Nodes",
    "Novita",
    "Pending",
    "Provider",
    "RunPod",
    "Scaleway",
    "SkywardError",
    "Spec",
    "TaskFailedError",
    "TaskIndeterminateError",
    "TensorDock",
    "VastAI",
    "Verda",
    "Vultr",
    "accelerators",
    "function",
    "gather",
    "instance_info",
    "is_head",
    "shard",
    "silent",
    "stderr",
    "stdout",
]


def __getattr__(name: str) -> object:
    match name:
        case "accelerators":
            value = importlib.import_module("skyward2.accelerators")
        case _ if name in RUNTIME:
            value = getattr(importlib.import_module("skyward2.runtime.api"), name)
        case _ if name in __all__:
            value = getattr(importlib.import_module("skyward2.sdk"), name)
        case _:
            raise AttributeError(f"module 'skyward2' has no attribute '{name}'")

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
