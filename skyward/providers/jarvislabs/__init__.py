"""Jarvis Labs GPU cloud provider for Skyward.

Uses the jarvislabs Python SDK (sync, dispatched via ThreadPoolExecutor).
Supports IN1, IN2, and EU1 regions with per-minute billing.

NOTE: Only config classes are imported at package level to avoid deps.

Environment Variables:
    JL_API_KEY: API token (required if not passed directly)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .provider import JarvisLabsProvider, JarvisLabsSpecific

from .config import JarvisLabs


def __getattr__(name: str) -> Any:
    if name == "JarvisLabsProvider":
        from .provider import JarvisLabsProvider

        return JarvisLabsProvider
    if name == "JarvisLabsSpecific":
        from .provider import JarvisLabsSpecific

        return JarvisLabsSpecific
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "JarvisLabs",
    "JarvisLabsProvider",
    "JarvisLabsSpecific",
]
