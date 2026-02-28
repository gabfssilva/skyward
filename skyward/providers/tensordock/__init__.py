"""TensorDock provider for Skyward.

NOTE: Only config classes are imported at package level to avoid deps.

Environment Variables:
    TENSORDOCK_API_KEY: API key (required if not passed directly)
    TENSORDOCK_API_TOKEN: API token (required if not passed directly)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .client import TensorDockClient, TensorDockError
    from .types import (
        AuthTestResponse,
        DeployResponse,
        HostnodeResponse,
        VmDetails,
        VmGetResponse,
        VmListResponse,
    )

from .config import TensorDock


def __getattr__(name: str) -> Any:
    if name in ("TensorDockClient", "TensorDockError"):
        from .client import TensorDockClient, TensorDockError
        if name == "TensorDockClient":
            return TensorDockClient
        return TensorDockError
    if name in (
        "AuthTestResponse",
        "DeployResponse",
        "HostnodeResponse",
        "VmDetails",
        "VmGetResponse",
        "VmListResponse",
    ):
        from .types import (
            AuthTestResponse,
            DeployResponse,
            HostnodeResponse,
            VmDetails,
            VmGetResponse,
            VmListResponse,
        )
        if name == "AuthTestResponse":
            return AuthTestResponse
        if name == "DeployResponse":
            return DeployResponse
        if name == "HostnodeResponse":
            return HostnodeResponse
        if name == "VmDetails":
            return VmDetails
        if name == "VmGetResponse":
            return VmGetResponse
        return VmListResponse
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TensorDock",
    "TensorDockClient",
    "TensorDockError",
    "AuthTestResponse",
    "DeployResponse",
    "HostnodeResponse",
    "VmDetails",
    "VmGetResponse",
    "VmListResponse",
]
