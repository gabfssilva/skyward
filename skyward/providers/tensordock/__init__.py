"""TensorDock provider for Skyward.

NOTE: Only config classes are imported at package level to avoid deps.

Environment Variables:
    TENSORDOCK_API_TOKEN: API token / Bearer token (required if not passed directly)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .client import TensorDockClient, TensorDockError
    from .types import (
        Location,
        LocationGpu,
        V2InstanceResponse,
    )

from .config import TensorDock


def __getattr__(name: str) -> Any:
    if name in ("TensorDockClient", "TensorDockError"):
        from .client import TensorDockClient, TensorDockError
        if name == "TensorDockClient":
            return TensorDockClient
        return TensorDockError
    if name in ("Location", "LocationGpu", "V2InstanceResponse"):
        from .types import Location, LocationGpu, V2InstanceResponse
        match name:
            case "Location":
                return Location
            case "LocationGpu":
                return LocationGpu
            case "V2InstanceResponse":
                return V2InstanceResponse
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TensorDock",
    "TensorDockClient",
    "TensorDockError",
    "Location",
    "LocationGpu",
    "V2InstanceResponse",
]
