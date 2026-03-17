"""Jarvis Labs provider configuration."""

from __future__ import annotations

import typing
from dataclasses import dataclass

from skyward.core.provider import ProviderConfig

if typing.TYPE_CHECKING:
    from skyward.providers.jarvislabs.provider import JarvisLabsProvider


@dataclass(frozen=True, slots=True)
class JarvisLabs(ProviderConfig):
    """Jarvis Labs GPU cloud configuration.

    Parameters
    ----------
    api_key
        API token. Falls back to JL_API_KEY env var.
    region
        Preferred region (IN1, IN2, EU1). If None, auto-selects
        from GPU availability.
    template
        Framework template for instances. Default: pytorch.
        Options: pytorch, tensorflow, jax, vm.
    storage_gb
        Disk storage per instance in GB. Default: 50.
        EU1 and VM template require minimum 100 GB.
    instance_timeout
        Auto-shutdown safety timeout in seconds. Default: 300.
    thread_pool_size
        Max worker threads for SDK calls. Default: 8.
    """

    api_key: str | None = None
    region: str | None = None
    template: str = "pytorch"
    storage_gb: int = 50
    instance_timeout: int = 300
    thread_pool_size: int = 8

    @property
    def type(self) -> str:
        return "jarvislabs"

    async def create_provider(self) -> JarvisLabsProvider:
        from skyward.providers.jarvislabs.provider import JarvisLabsProvider

        return await JarvisLabsProvider.create(self)
