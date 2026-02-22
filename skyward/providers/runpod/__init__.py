"""RunPod GPU Pods provider for Skyward.

NOTE: Only config classes are imported at package level to avoid deps.

Environment Variables:
    RUNPOD_API_KEY: API key (required if not passed directly)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .client import RunPodClient, RunPodError

from .config import CloudType, ClusterMode, RunPod


def __getattr__(name: str) -> Any:
    if name in ("RunPodClient", "RunPodError"):
        from .client import RunPodClient, RunPodError
        if name == "RunPodClient":
            return RunPodClient
        return RunPodError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CloudType",
    "ClusterMode",
    "RunPod",
    "RunPodClient",
    "RunPodError",
]
