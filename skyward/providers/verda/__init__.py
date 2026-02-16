"""Verda Cloud provider for Skyward v2.

Verda (formerly DataCrunch) is a cloud provider specializing in AI and ML services.
Features dedicated GPU instances, GPU clusters, and serverless inference.

NOTE: Only config classes are imported at package level to avoid deps.
For handlers and modules, import explicitly:

    from skyward.providers.verda.handler import VerdaHandler
    from skyward.providers.verda.client import VerdaClient

Environment Variables:
    VERDA_CLIENT_ID: API client ID (required if not passed directly)
    VERDA_CLIENT_SECRET: API client secret (required if not passed directly)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .client import VerdaClient, VerdaError
    from .provider import VerdaCloudProvider

from .config import Verda

__all__ = [
    "Verda",
    "VerdaClient",
    "VerdaError",
    "VerdaCloudProvider"
]
