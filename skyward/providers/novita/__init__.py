"""Novita.ai provider for Skyward.

NOTE: Only config classes are imported at package level to avoid deps.

Environment Variables:
    NOVITA_API_KEY: API key (required if not passed directly)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .client import NovitaClient, NovitaError, get_api_key
    from .types import InstanceResponse, ProductResponse

from .config import Novita


def __getattr__(name: str) -> Any:
    if name in ("NovitaClient", "NovitaError", "get_api_key"):
        from .client import NovitaClient, NovitaError, get_api_key
        if name == "NovitaClient":
            return NovitaClient
        if name == "NovitaError":
            return NovitaError
        return get_api_key
    if name in ("InstanceResponse", "ProductResponse"):
        from .types import InstanceResponse, ProductResponse
        if name == "InstanceResponse":
            return InstanceResponse
        return ProductResponse
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Novita",
    "NovitaClient",
    "NovitaError",
    "get_api_key",
    "InstanceResponse",
    "ProductResponse",
]
