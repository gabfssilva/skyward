"""Scaleway provider for Skyward.

NOTE: Only config classes are imported at package level to avoid deps.

Environment Variables:
    SCW_SECRET_KEY: API secret key (required if not passed directly)
    SCW_DEFAULT_PROJECT_ID: Project ID (required if not passed directly)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .client import ScalewayClient, ScalewayError, get_project_id, get_secret_key
    from .provider import ScalewayProvider, ScalewaySpecific
    from .types import (
        ImageResponse,
        IpResponse,
        ServerImageResponse,
        ServerIpResponse,
        ServerResponse,
        ServerTypeGpuInfo,
        ServerTypeResponse,
        SSHKeyResponse,
    )

from .config import Scaleway


def __getattr__(name: str) -> Any:
    if name in ("ScalewayClient", "ScalewayError", "get_secret_key", "get_project_id"):
        from .client import ScalewayClient, ScalewayError, get_project_id, get_secret_key

        _map = {
            "ScalewayClient": ScalewayClient,
            "ScalewayError": ScalewayError,
            "get_secret_key": get_secret_key,
            "get_project_id": get_project_id,
        }
        return _map[name]
    if name in ("ScalewayProvider", "ScalewaySpecific"):
        from .provider import ScalewayProvider, ScalewaySpecific

        return {"ScalewayProvider": ScalewayProvider, "ScalewaySpecific": ScalewaySpecific}[name]
    if name in (
        "ImageResponse", "IpResponse", "SSHKeyResponse", "ServerImageResponse",
        "ServerIpResponse", "ServerResponse", "ServerTypeGpuInfo", "ServerTypeResponse",
    ):
        from . import types

        return getattr(types, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Scaleway",
    "ScalewayClient",
    "ScalewayError",
    "ScalewayProvider",
    "ScalewaySpecific",
    "get_secret_key",
    "get_project_id",
    "ImageResponse",
    "IpResponse",
    "SSHKeyResponse",
    "ServerImageResponse",
    "ServerIpResponse",
    "ServerResponse",
    "ServerTypeGpuInfo",
    "ServerTypeResponse",
]
