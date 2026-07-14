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

__all__ = [
    "AWS",
    "GCP",
    "Accelerator",
    "Compute",
    "Container",
    "Group",
    "Hyperstack",
    "Image",
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
]


def __getattr__(name: str) -> object:
    match name:
        case "accelerators":
            value = importlib.import_module("skyward2.accelerators")
        case _ if name in __all__:
            value = getattr(importlib.import_module("skyward2.sdk"), name)
        case _:
            raise AttributeError(f"module 'skyward2' has no attribute '{name}'")

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
