"""Verda provider configuration.

Immutable configuration dataclass for Verda provider.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

from skyward.api.provider import ProviderConfig

if typing.TYPE_CHECKING:
    from skyward.providers.verda.provider import VerdaCloudProvider

# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class Verda(ProviderConfig[VerdaCloudProvider]):
    """Verda Cloud provider configuration.

    SSH keys are automatically detected from ~/.ssh/id_ed25519.pub or
    ~/.ssh/id_rsa.pub and registered on Verda if needed.

    Features:
        - Auto-region discovery: If the requested region doesn't have capacity,
          automatically finds another region with availability.
        - Spot instances: Supports spot pricing for cost savings.

    Example:
        >>> from skyward.providers.verda import Verda
        >>> config = Verda(region="FIN-01")

    Args:
        region: Preferred region (e.g., "FIN-01"). Default: FIN-01.
        client_id: Verda client ID. Falls back to VERDA_CLIENT_ID env var.
        client_secret: Verda client secret. Falls back to VERDA_CLIENT_SECRET env var.
        ssh_key_id: Specific SSH key ID to use (optional).
        instance_timeout: Safety timeout in seconds. Default: 300.
    """

    region: str = "FIN-01"
    client_id: str | None = None
    client_secret: str | None = None
    ssh_key_id: str | None = None
    instance_timeout: int = 300
    request_timeout: int = 30

    async def create_provider(self) -> VerdaCloudProvider:
        from skyward.providers.verda.provider import VerdaCloudProvider
        return await VerdaCloudProvider.create(self)
