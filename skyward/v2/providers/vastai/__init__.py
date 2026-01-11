"""Vast.ai provider for Skyward v2.

Vast.ai is a GPU marketplace where hosts offer their machines for rent.
Instances are Docker containers with varying reliability and specifications.

NOTE: Only config classes are imported at package level to avoid deps.
For handlers and modules, import explicitly:

    from skyward.v2.providers.vastai.handler import VastAIHandler
    from skyward.v2.providers.vastai.client import VastAIClient

Environment Variables:
    VAST_API_KEY: API key (required if not passed directly)
"""

# Only config - no heavy dependencies
from .config import VastAI


# Lazy imports for backward compatibility
def __getattr__(name: str):
    if name == "VastAIHandler":
        from .handler import VastAIHandler
        return VastAIHandler
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
    if name == "VastAIModule":
        from injector import Module, provider, singleton

        class VastAIModule(Module):
            """DI module for Vast.ai provider."""

            @singleton
            @provider
            def provide_config(self) -> VastAI:
                """Provide default Vast.ai configuration."""
                return VastAI()

        return VastAIModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Config (always available)
    "VastAI",
    # Lazy (loaded on demand)
    "VastAIClient",
    "VastAIError",
    "get_api_key",
    "VastAIHandler",
    "VastAIClusterState",
    "OfferResponse",
    "InstanceResponse",
    "SSHKeyResponse",
    "VastAIModule",
]
