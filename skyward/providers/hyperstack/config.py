"""Hyperstack provider configuration.

Immutable configuration dataclass for Hyperstack InfraHub provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from skyward.core.provider import ProviderConfig

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
    image
        OS image name override. When None, auto-selects the newest
        Ubuntu + CUDA image available in the region.
    network_optimised
        Require network-optimised environments with SR-IOV support
        (up to 350 Gbps inter-VM bandwidth). Only available in
        certain regions. Default False.
    network_optimised_regions
        Regions known to support network-optimised environments.
        Override when Hyperstack adds new SR-IOV regions.
    object_storage_region
        Region for S3-compatible object storage (volume mounts).
    object_storage_endpoint
        Endpoint URL for S3-compatible object storage.
    instance_timeout
        Auto-shutdown safety timeout in seconds.
    request_timeout
        HTTP request timeout in seconds.
    """

    api_key: str | None = None
    region: str | tuple[str, ...] | None = None
    image: str | None = None
    network_optimised: bool = False
    network_optimised_regions: tuple[str, ...] = ("CANADA-1", "US-1")
    object_storage_region: str = "CANADA-1"
    object_storage_endpoint: str = "https://ca1.obj.nexgencloud.io"
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
