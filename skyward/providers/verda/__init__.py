"""Verda Cloud provider for Skyward.

Verda (formerly DataCrunch) is a cloud provider specializing in AI and ML services.
Features dedicated GPU instances, GPU clusters, and serverless inference.

NOTE: Only config classes are imported at package level to avoid deps.

Environment Variables:
    VERDA_CLIENT_ID: API client ID (required if not passed directly)
    VERDA_CLIENT_SECRET: API client secret (required if not passed directly)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .client import VerdaClient, VerdaError
    from .provider import VerdaProvider

from .config import Verda


def __getattr__(name: str) -> Any:
    if name in ("VerdaClient", "VerdaError"):
        from .client import VerdaClient, VerdaError

        return {"VerdaClient": VerdaClient, "VerdaError": VerdaError}[name]
    if name == "VerdaProvider":
        from .provider import VerdaProvider

        return VerdaProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Verda",
    "VerdaClient",
    "VerdaError",
    "VerdaProvider",
]
