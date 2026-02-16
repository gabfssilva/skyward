"""AWS Provider for Skyward.

NOTE: Only config classes are imported at package level to avoid SDK deps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .clients import Client

from .config import AWS, AllocationStrategy


def __getattr__(name: str) -> Any:
    if name == "Client":
        from .clients import Client
        return Client
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AWS",
    "AllocationStrategy",
    "Client",
]
