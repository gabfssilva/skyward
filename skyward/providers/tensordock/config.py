"""TensorDock provider configuration."""

from __future__ import annotations

import typing
from dataclasses import dataclass

from skyward.api.provider import ProviderConfig

if typing.TYPE_CHECKING:
    from typing import Literal

    from skyward.providers.tensordock.provider import TensorDockProvider


@dataclass(frozen=True, slots=True)
class TensorDock(ProviderConfig):
    """TensorDock GPU marketplace configuration (v2 API).

    Parameters
    ----------
    api_key
        API key. Falls back to TENSORDOCK_API_KEY env var.
    api_token
        API token (used as Bearer token for v2 API).
        Falls back to TENSORDOCK_API_TOKEN env var.
    location
        Country filter (e.g., "United States", "Germany"). None means global.
    storage_gb
        Disk storage per VM in GB. Minimum 100.
    operating_system
        OS image ID for deployment (e.g., "ubuntu2404", "ubuntu2204").
    instance_timeout
        Auto-shutdown in seconds. Default: 300.
    request_timeout
        HTTP request timeout in seconds. Default: 30.
    min_ram_gb
        Minimum RAM per VM in GB. None for provider default (16).
    min_vcpus
        Minimum vCPUs per VM. None for provider default (4).
    """

    api_key: str | None = None
    api_token: str | None = None
    location: str | None = None
    tier: Literal[0, 1, 2, 3, 4] | None = None
    storage_gb: int = 100
    operating_system: str = "ubuntu2404"
    instance_timeout: int = 300
    request_timeout: int = 120
    min_ram_gb: int | None = None
    min_vcpus: int | None = None

    @property
    def type(self) -> str:
        return "tensordock"

    async def create_provider(self) -> TensorDockProvider:
        from skyward.providers.tensordock.provider import TensorDockProvider

        return await TensorDockProvider.create(self)
