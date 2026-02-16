"""Vast.ai provider for Skyward.

NOTE: Only config classes are imported at package level to avoid deps.

Environment Variables:
    VAST_API_KEY: API key (required if not passed directly)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .client import VastAIClient, VastAIError, get_api_key
    from .types import InstanceResponse, OfferResponse, SSHKeyResponse

from .config import VastAI


def __getattr__(name: str) -> Any:
    if name in ("VastAIClient", "VastAIError", "get_api_key"):
        from .client import VastAIClient, VastAIError, get_api_key
        if name == "VastAIClient":
            return VastAIClient
        if name == "VastAIError":
            return VastAIError
        return get_api_key
    if name in ("OfferResponse", "InstanceResponse", "SSHKeyResponse"):
        from .types import InstanceResponse, OfferResponse, SSHKeyResponse
        if name == "OfferResponse":
            return OfferResponse
        if name == "InstanceResponse":
            return InstanceResponse
        return SSHKeyResponse
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "VastAI",
    "VastAIClient",
    "VastAIError",
    "get_api_key",
    "OfferResponse",
    "InstanceResponse",
    "SSHKeyResponse",
]
