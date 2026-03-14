"""Vultr GPU cloud provider for Skyward.

NOTE: Only config classes are imported at package level to avoid deps.

Environment Variables:
    VULTR_API_KEY: API key (required if not passed directly)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .client import VultrClient, VultrError

from .config import Vultr


def __getattr__(name: str) -> Any:
    if name in ("VultrClient", "VultrError"):
        from .client import VultrClient, VultrError
        if name == "VultrClient":
            return VultrClient
        return VultrError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Vultr",
    "VultrClient",
    "VultrError",
]
