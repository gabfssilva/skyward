"""Vultr provider configuration."""

from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Literal

from skyward.core.provider import ProviderConfig

if typing.TYPE_CHECKING:
    from skyward.providers.vultr.provider import VultrProvider

type VultrMode = Literal["cloud", "bare-metal"]


@dataclass(frozen=True, slots=True)
class Vultr(ProviderConfig):
    """Vultr GPU cloud provider configuration.

    Supports two modes: Cloud GPU (virtual instances with ``vcg-*`` plans)
    and Bare Metal (dedicated servers with ``vbm-*`` plans).

    Parameters
    ----------
    api_key
        API key. Falls back to ``VULTR_API_KEY`` env var.
    mode
        Instance mode. ``"cloud"`` for virtual GPU instances (default),
        ``"bare-metal"`` for dedicated physical servers.
    region
        Preferred region ID (e.g., ``"ewr"``, ``"ord"``). Default: ``"ewr"``.
    os_id
        OS image ID. Default: ``2284`` (Ubuntu 24.04).
    instance_timeout
        Safety timeout in seconds. Default: ``300``.
    request_timeout
        HTTP request timeout in seconds. Default: ``30``.
    """

    api_key: str | None = None
    mode: VultrMode = "cloud"
    region: str = "ewr"
    os_id: int = 2284
    instance_timeout: int = 300
    request_timeout: int = 30

    async def create_provider(self) -> VultrProvider:
        from skyward.providers.vultr.provider import VultrProvider
        return await VultrProvider.create(self)

    @property
    def type(self) -> str: return "vultr"
