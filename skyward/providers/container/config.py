from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from skyward.api.provider import ProviderConfig

if TYPE_CHECKING:
    from skyward.providers.container.provider import ContainerProvider

_DEFAULT_IMAGE = "ubuntu:24.04"


@dataclass(frozen=True, slots=True)
class Container(ProviderConfig):
    """Container provider configuration.

    Runs compute nodes as local containers with SSH access.
    Supports Docker, podman, nerdctl, and Apple's container CLI
    via the configurable ``binary`` field.

    Useful for local development and CI testing without cloud costs.

    Example:
        >>> import skyward as sky
        >>> with sky.pool(provider=sky.Container(), nodes=2) as p:
        ...     result = train(data) >> sky
    """

    image: str = _DEFAULT_IMAGE
    ssh_user: str = "root"
    binary: str = "docker"

    async def create_provider(self) -> ContainerProvider:
        from skyward.providers.container.provider import ContainerProvider
        return await ContainerProvider.create(self)

    @property
    def type(self) -> str: return "container"
