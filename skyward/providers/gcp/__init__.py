"""GCP Compute Engine provider for Skyward.

Google Cloud Platform provider using Compute Engine instances.
Supports GPU-accelerated VMs, spot/preemptible instances, and
custom machine types.

NOTE: Only config classes are imported at package level to avoid deps.
For provider implementation, import explicitly:

    from skyward.providers.gcp.provider import GCPProvider

Environment Variables:
    GOOGLE_CLOUD_PROJECT: GCP project ID (required if not passed directly)
    GOOGLE_APPLICATION_CREDENTIALS: Path to service account key (optional)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .provider import GCPProvider

from .config import GCP

__all__ = [
    "GCP",
    "GCPProvider",
]
