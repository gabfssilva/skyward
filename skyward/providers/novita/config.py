"""Novita.ai provider configuration.

Immutable configuration dataclass for Novita.ai provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from skyward.containers import DockerImage
from skyward.core.provider import ProviderConfig

if TYPE_CHECKING:
    from skyward.api.spec import Options
    from skyward.providers.novita.provider import NovitaProvider


@dataclass(frozen=True, slots=True)
class Novita(ProviderConfig):
    """Novita.ai provider configuration.

    Novita.ai provides GPU cloud instances with SSH access.
    Instances are Docker containers with configurable GPU count,
    root filesystem size, and startup commands.

    SSH access is provided by the Novita proxy — no openssh-server
    or key injection is needed inside the container.

    Parameters
    ----------
    api_key
        Novita.ai API key. Falls back to ``NOVITA_API_KEY`` env var.
    cluster_id
        Target cluster/region ID. ``None`` for auto-selection.
    rootfs_size
        Root filesystem size in GB.
    docker_image
        Base Docker image for containers. ``None`` uses the default
        CUDA runtime image.
    min_cuda_version
        Minimum CUDA version requirement (e.g., ``"12.4"``).
    request_timeout
        HTTP request timeout in seconds.

    Examples
    --------
    >>> from skyward.providers.novita import Novita
    >>> config = Novita(rootfs_size=100, min_cuda_version="12.4")
    """

    api_key: str | None = None
    cluster_id: str | None = None
    rootfs_size: int = 50
    docker_image: DockerImage | None = None
    min_cuda_version: str | None = None
    request_timeout: int = 30

    async def create_provider(self) -> NovitaProvider:
        from skyward.providers.novita.provider import NovitaProvider
        return await NovitaProvider.create(self)

    @property
    def type(self) -> str: return "novita"

    def default_options(self) -> Options:
        from skyward.api.spec import Options
        return Options(
            provision_timeout=600,
            ssh_timeout=600,
            bootstrap_timeout=600,
            cluster=False,
        )
