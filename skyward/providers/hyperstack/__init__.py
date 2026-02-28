"""Hyperstack provider for Skyward.

NOTE: Only config classes are imported at package level to avoid deps.

Environment Variables:
    HYPERSTACK_API_KEY: API key (required if not passed directly)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .client import HyperstackAuth, HyperstackClient, HyperstackError, get_api_key
    from .provider import HyperstackProvider, HyperstackSpecific
    from .types import (
        CreateVMPayload,
        CreateVMResponse,
        EnvironmentResponse,
        FlavorResponse,
        ImageResponse,
        KeypairResponse,
        PricebookEntry,
        VMResponse,
    )

from .config import Hyperstack


def __getattr__(name: str) -> Any:
    if name in ("HyperstackClient", "HyperstackError", "HyperstackAuth", "get_api_key"):
        from .client import HyperstackAuth, HyperstackClient, HyperstackError, get_api_key

        _map = {
            "HyperstackClient": HyperstackClient,
            "HyperstackError": HyperstackError,
            "HyperstackAuth": HyperstackAuth,
            "get_api_key": get_api_key,
        }
        return _map[name]
    if name in ("HyperstackProvider", "HyperstackSpecific"):
        from .provider import HyperstackProvider, HyperstackSpecific

        return {"HyperstackProvider": HyperstackProvider, "HyperstackSpecific": HyperstackSpecific}[name]
    if name in (
        "FlavorResponse", "ImageResponse", "EnvironmentResponse",
        "KeypairResponse", "VMResponse", "CreateVMPayload",
        "CreateVMResponse", "PricebookEntry",
    ):
        from . import types

        return getattr(types, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Hyperstack",
    "HyperstackClient",
    "HyperstackError",
    "HyperstackAuth",
    "HyperstackProvider",
    "HyperstackSpecific",
    "get_api_key",
    "FlavorResponse",
    "ImageResponse",
    "EnvironmentResponse",
    "KeypairResponse",
    "VMResponse",
    "CreateVMPayload",
    "CreateVMResponse",
    "PricebookEntry",
]
