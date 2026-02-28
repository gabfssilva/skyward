"""Lambda Cloud provider for Skyward.

NOTE: Only config classes are imported at package level to avoid deps.

Environment Variables:
    LAMBDA_API_KEY: API key (required if not passed directly)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .client import LambdaClient, LambdaError, get_api_key
    from .types import InstanceResponse, InstanceTypeEntry, SSHKeyResponse

from .config import Lambda


def __getattr__(name: str) -> Any:
    if name in ("LambdaClient", "LambdaError", "get_api_key"):
        from .client import LambdaClient, LambdaError, get_api_key

        match name:
            case "LambdaClient":
                return LambdaClient
            case "LambdaError":
                return LambdaError
            case "get_api_key":
                return get_api_key
    if name in ("InstanceResponse", "InstanceTypeEntry", "SSHKeyResponse"):
        from .types import InstanceResponse, InstanceTypeEntry, SSHKeyResponse

        match name:
            case "InstanceResponse":
                return InstanceResponse
            case "InstanceTypeEntry":
                return InstanceTypeEntry
            case "SSHKeyResponse":
                return SSHKeyResponse
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Lambda",
    "LambdaClient",
    "LambdaError",
    "get_api_key",
    "InstanceResponse",
    "InstanceTypeEntry",
    "SSHKeyResponse",
]
