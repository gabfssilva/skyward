"""Vast.ai provider configuration.

Immutable configuration dataclass for Vast.ai provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from skyward.containers import DockerImage
from skyward.core.provider import ProviderConfig

if TYPE_CHECKING:
    from skyward.providers.vastai.provider import VastAIProvider

# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class VastAI(ProviderConfig):
    """Vast.ai provider configuration.

    Vast.ai is a GPU marketplace with dynamic offers from various hosts.
    Unlike traditional cloud providers, instances are Docker containers
    running on marketplace hosts with varying reliability.

    SSH keys are automatically detected from ~/.ssh/id_ed25519.pub or
    ~/.ssh/id_rsa.pub and registered on Vast.ai if needed.

    Example:
        >>> from skyward.providers.vastai import VastAI
        >>> config = VastAI(min_reliability=0.95, geolocation="US")

    Args:
        api_key: Vast.ai API key. Falls back to VAST_API_KEY env var.
        min_reliability: Minimum host reliability score (0.0-1.0).
        min_cuda: Minimum CUDA version (e.g., 12.0). Filters out old/broken hosts.
        geolocation: Preferred region/country code (e.g., "US", "EU").
        bid_multiplier: For spot pricing, multiply min bid by this.
        instance_timeout: Auto-shutdown in seconds (safety timeout).
        docker_image: Base Docker image for containers.
        disk_gb: Disk space in GB.
        overlay_timeout: Timeout for overlay operations in seconds.
        require_direct_port: Only select offers with direct port access (no SSH proxy).
        verified_only: Only select offers from verified hosts (default True).
        min_inet_down: Minimum download speed in Mbps. None disables filter.
        min_inet_up: Minimum upload speed in Mbps. None disables filter.
    """

    api_key: str | None = None
    min_reliability: float = 0.95
    verified_only: bool = True
    min_cuda: float = 12.0
    geolocation: str | None = None
    bid_multiplier: float = 1.2
    instance_timeout: int = 300
    request_timeout: int = 30
    docker_image: DockerImage | None = None
    disk_gb: int = 100
    overlay_timeout: int = 120
    require_direct_port: bool = False
    min_inet_down: float | None = None
    min_inet_up: float | None = None

    async def create_provider(self) -> VastAIProvider:
        from skyward.providers.vastai.provider import VastAIProvider
        return await VastAIProvider.create(self)

    @property
    def type(self) -> str: return "vastai"

    def default_options(self) -> None:
        return None
