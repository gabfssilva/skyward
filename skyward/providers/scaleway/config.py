"""Scaleway provider configuration.

Immutable configuration dataclass for Scaleway GPU instances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from skyward.core.provider import ProviderConfig

if TYPE_CHECKING:
    from skyward.providers.scaleway.provider import ScalewayProvider


@dataclass(frozen=True, slots=True)
class Scaleway(ProviderConfig):
    """Scaleway GPU cloud provider configuration.

    Scaleway provides GPU instances (L4, L40S, H100, B300) in European
    zones. Auth uses a secret key passed via ``X-Auth-Token`` header.

    Parameters
    ----------
    secret_key
        Scaleway secret key (UUID). Falls back to SCW_SECRET_KEY env var.
    project_id
        Scaleway project ID (UUID). Falls back to SCW_DEFAULT_PROJECT_ID env var.
    zone
        Availability zone. ``None`` searches all GPU zones automatically.
    image
        OS image UUID override. When None, auto-selects the newest
        Ubuntu GPU image available.
    instance_timeout
        Auto-shutdown safety timeout in seconds. Default: 300.
    request_timeout
        HTTP request timeout in seconds. Default: 30.
    """

    secret_key: str | None = None
    project_id: str | None = None
    zone: str | None = None
    image: str | None = None
    instance_timeout: int = 300
    request_timeout: int = 30

    @property
    def type(self) -> str:
        return "scaleway"

    async def create_provider(self) -> ScalewayProvider:
        from skyward.providers.scaleway.provider import ScalewayProvider

        return await ScalewayProvider.create(self)
