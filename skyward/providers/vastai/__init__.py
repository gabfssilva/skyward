"""Vast.ai provider for Skyward v2.

Vast.ai is a GPU marketplace where hosts offer their machines for rent.
Instances are Docker containers with varying reliability and specifications.

NOTE: Only config classes are imported at package level to avoid deps.
For handlers and modules, import explicitly:

    from skyward.providers.vastai.handler import VastAIHandler
    from skyward.providers.vastai.client import VastAIClient

Environment Variables:
    VAST_API_KEY: API key (required if not passed directly)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .client import VastAIClient, VastAIError, get_api_key
    from .handler import vastai_provider_actor
    from .state import VastAIClusterState
    from .types import InstanceResponse, OfferResponse, SSHKeyResponse

# Only config - no heavy dependencies
from .config import VastAI


def __getattr__(name: str) -> Any:
    if name == "vastai_provider_actor":
        from .handler import vastai_provider_actor
        return vastai_provider_actor
    if name in ("VastAIClient", "VastAIError", "get_api_key"):
        from .client import VastAIClient, VastAIError, get_api_key
        if name == "VastAIClient":
            return VastAIClient
        if name == "VastAIError":
            return VastAIError
        return get_api_key
    if name == "VastAIClusterState":
        from .state import VastAIClusterState
        return VastAIClusterState
    if name in ("OfferResponse", "InstanceResponse", "SSHKeyResponse"):
        from .types import InstanceResponse, OfferResponse, SSHKeyResponse
        if name == "OfferResponse":
            return OfferResponse
        if name == "InstanceResponse":
            return InstanceResponse
        return SSHKeyResponse
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Config (always available)
    "VastAI",
    # Lazy (loaded on demand)
    "VastAIClient",
    "VastAIError",
    "get_api_key",
    "vastai_provider_actor",
    "VastAIClusterState",
    "OfferResponse",
    "InstanceResponse",
    "SSHKeyResponse",
]
