"""GCP Compute Engine provider for Skyward.

Google Cloud Platform provider using Compute Engine instances.
Supports GPU-accelerated VMs, spot/preemptible instances, and
custom machine types.

NOTE: Only config classes are imported at package level to avoid deps.

Environment Variables:
    GOOGLE_CLOUD_PROJECT: GCP project ID (required if not passed directly)
    GOOGLE_APPLICATION_CREDENTIALS: Path to service account key (optional)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .provider import GCPProvider
    from .types import GCPSpecific, ResolvedMachine

from .config import GCP


def __getattr__(name: str) -> Any:
    if name in ("GCPProvider",):
        from .provider import GCPProvider

        return GCPProvider
    if name in ("GCPSpecific", "ResolvedMachine"):
        from .types import GCPSpecific, ResolvedMachine

        return {"GCPSpecific": GCPSpecific, "ResolvedMachine": ResolvedMachine}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GCP",
    "GCPProvider",
    "GCPSpecific",
    "ResolvedMachine",
]
