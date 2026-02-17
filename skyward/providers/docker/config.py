from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from skyward.api.provider import ProviderConfig

if TYPE_CHECKING:
    from skyward.providers.docker.provider import DockerCloudProvider

_DEFAULT_IMAGE = "ubuntu:24.04"


@dataclass(frozen=True, slots=True)
class Docker(ProviderConfig):
    """Docker provider configuration.

    Runs compute nodes as local Docker containers with SSH access.
    Useful for local development and CI testing without cloud costs.

    The provider creates a bridge network per cluster and launches
    containers with Dropbear SSH, injecting the local SSH public key.

    Example:
        >>> import skyward as sky
        >>> with sky.pool(provider=sky.Docker(), nodes=2) as p:
        ...     result = train(data) >> sky
    """

    image: str = _DEFAULT_IMAGE
    ssh_user: str = "root"

    async def create_provider(self) -> DockerCloudProvider:
        from skyward.providers.docker.provider import DockerCloudProvider
        return await DockerCloudProvider.create(self)

    @property
    def type(self) -> str: return "docker"
