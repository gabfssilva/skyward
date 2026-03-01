"""Hyperstack provider configuration.

Immutable configuration dataclass for Hyperstack InfraHub provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from skyward.api.provider import ProviderConfig

if TYPE_CHECKING:
    from skyward.providers.hyperstack.provider import HyperstackProvider


@dataclass(frozen=True, slots=True)
class Hyperstack(ProviderConfig):
    """Hyperstack GPU cloud provider configuration.

    Hyperstack provides bare-metal GPU instances via their InfraHub API.
    Resources are organized into environments that group VMs, keypairs,
    and volumes within a region.

    Parameters
    ----------
    api_key
        Hyperstack API key. Falls back to HYPERSTACK_API_KEY env var.
    region
        Deployment region(s). A single string, a tuple of strings,
        or None to search all regions. Examples: ``"CANADA-1"``,
        ``("CANADA-1", "NORWAY-1")``, ``None``.
    instance_timeout
        Auto-shutdown safety timeout in seconds.
    request_timeout
        HTTP request timeout in seconds.
    """

    api_key: str | None = None
    region: str | tuple[str, ...] | None = None
    instance_timeout: int = 300
    request_timeout: int = 30
    teardown_timeout: int = 120
    teardown_poll_interval: float = 2.0

    @property
    def type(self) -> str:
        return "hyperstack"

    async def create_provider(self) -> HyperstackProvider:
        from skyward.providers.hyperstack.provider import HyperstackProvider

        return await HyperstackProvider.create(self)
