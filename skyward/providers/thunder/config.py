from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Literal

from skyward.api.provider import ProviderConfig

if typing.TYPE_CHECKING:
    from skyward.providers.thunder.provider import ThunderProvider


@dataclass(frozen=True, slots=True)
class ThunderCompute(ProviderConfig):
    """Thunder Compute provider configuration.

    Parameters
    ----------
    api_token
        API token. Falls back to TNR_API_TOKEN env var, then ~/.thunder/token.
    mode
        Instance mode. "production" for full CUDA + multi-GPU,
        "prototyping" for cheaper limited CUDA.
    template
        Instance template. "base", "ollama", "comfy-ui", or a snapshot ID.
    disk_size_gb
        Disk size in GB.
    cpu_cores
        Number of CPU cores.
    request_timeout
        HTTP request timeout in seconds.
    """

    api_token: str | None = None
    mode: Literal["prototyping", "production"] = "production"
    template: str = "base"
    disk_size_gb: int = 100
    cpu_cores: int = 18
    request_timeout: int = 30

    async def create_provider(self) -> ThunderProvider:
        from skyward.providers.thunder.provider import ThunderProvider

        return await ThunderProvider.create(self)

    @property
    def type(self) -> str:
        return "thunder"
