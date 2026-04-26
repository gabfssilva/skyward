"""Verda provider configuration.

Immutable configuration dataclass for Verda provider.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

from skyward.core.provider import ProviderConfig

if typing.TYPE_CHECKING:
    from skyward.providers.verda.provider import VerdaProvider

# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class Verda(ProviderConfig):
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
        >>> Verda(cuda="12.8")  # pin a different CUDA build
        >>> Verda(image="ubuntu-22.04-cuda-12.4-docker")  # full override

    Args:
        region: Preferred region (e.g., "FIN-01"). Default: FIN-01.
        client_id: Verda client ID. Falls back to VERDA_CLIENT_ID env var.
        client_secret: Verda client secret. Falls back to VERDA_CLIENT_SECRET env var.
        ssh_key_id: Specific SSH key ID to use (optional).
        image: Full Verda image name to use verbatim. Bypasses the CUDA template.
        cuda: Exact CUDA version substituted into ``ubuntu-24.04-cuda-{cuda}-open``.
            Ignored when ``image`` is set. Default: ``"13.0"``.
        instance_timeout: Safety timeout in seconds. Default: 300.
    """

    region: str = "FIN-01"
    client_id: str | None = None
    client_secret: str | None = None
    ssh_key_id: str | None = None
    image: str | None = None
    cuda: str = "13.0"
    instance_timeout: int = 300
    request_timeout: int = 30

    @property
    def type(self) -> str: return "verda"

    def default_options(self) -> None:
        return None

    async def create_provider(self) -> VerdaProvider:
        from skyward.providers.verda.provider import VerdaProvider
        return await VerdaProvider.create(self)
