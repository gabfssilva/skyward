"""Massed Compute provider configuration.

Immutable configuration dataclass for Massed Compute provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from skyward.core.provider import ProviderConfig

if TYPE_CHECKING:
    from skyward.api.spec import Options
    from skyward.providers.massed_compute.provider import MassedComputeProvider


@dataclass(frozen=True, slots=True)
class MassedCompute(ProviderConfig):
    """Massed Compute GPU cloud provider.

    Bare-metal GPU instances with SSH access, spot and on-demand
    pricing. Supports A30 through H200 NVL and RTX PRO 6000
    Blackwell GPUs across US regions.

    Parameters
    ----------
    api_key
        API key. Falls back to ``MASSED_API_KEY`` env var.
    image_id
        OS image ID. Default ``184`` (Ubuntu Server 24.04).
    request_timeout
        HTTP request timeout in seconds.

    Examples
    --------
    >>> from skyward.providers.massed_compute import MassedCompute
    >>> config = MassedCompute()
    """

    api_key: str | None = None
    image_id: int = 184
    request_timeout: int = 30

    async def create_provider(self) -> MassedComputeProvider:
        from skyward.providers.massed_compute.provider import MassedComputeProvider
        return await MassedComputeProvider.create(self)

    @property
    def type(self) -> str: return "massed_compute"

    def default_options(self) -> Options:
        from skyward.api.spec import Options
        return Options(
            provision_timeout=600,
            ssh_timeout=600,
            bootstrap_timeout=600,
            cluster=False,
        )
