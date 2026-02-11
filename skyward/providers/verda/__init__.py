"""Verda Cloud provider for Skyward v2.

Verda (formerly DataCrunch) is a cloud provider specializing in AI and ML services.
Features dedicated GPU instances, GPU clusters, and serverless inference.

NOTE: Only config classes are imported at package level to avoid deps.
For handlers and modules, import explicitly:

    from skyward.providers.verda.handler import VerdaHandler
    from skyward.providers.verda.client import VerdaClient

Environment Variables:
    VERDA_CLIENT_ID: API client ID (required if not passed directly)
    VERDA_CLIENT_SECRET: API client secret (required if not passed directly)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .client import VerdaClient, VerdaError
    from .handler import verda_provider_actor
    from .state import VerdaClusterState

# Only config - no heavy dependencies
from .config import Verda


def __getattr__(name: str) -> Any:
    if name == "verda_provider_actor":
        from .handler import verda_provider_actor
        return verda_provider_actor
    if name in ("VerdaClient", "VerdaError"):
        from .client import VerdaClient, VerdaError
        if name == "VerdaClient":
            return VerdaClient
        return VerdaError
    if name == "VerdaClusterState":
        from .state import VerdaClusterState
        return VerdaClusterState
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Config (always available)
    "Verda",
    # Lazy (loaded on demand)
    "VerdaClient",
    "VerdaError",
    "verda_provider_actor",
    "VerdaClusterState",
]
