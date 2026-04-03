"""Massed Compute provider for Skyward.

NOTE: Only config classes are imported at package level to avoid deps.

Environment Variables:
    MASSED_API_KEY: API key (required if not passed directly)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .client import MassedComputeClient, MassedComputeError, get_api_key
    from .types import InstanceResponse, InventoryItem

from .config import MassedCompute


def __getattr__(name: str) -> Any:
    if name in ("MassedComputeClient", "MassedComputeError", "get_api_key"):
        from .client import MassedComputeClient, MassedComputeError, get_api_key
        match name:
            case "MassedComputeClient":
                return MassedComputeClient
            case "MassedComputeError":
                return MassedComputeError
            case "get_api_key":
                return get_api_key
    if name in ("InstanceResponse", "InventoryItem"):
        from .types import InstanceResponse, InventoryItem
        match name:
            case "InstanceResponse":
                return InstanceResponse
            case "InventoryItem":
                return InventoryItem
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MassedCompute",
    "MassedComputeClient",
    "MassedComputeError",
    "get_api_key",
    "InstanceResponse",
    "InventoryItem",
]
