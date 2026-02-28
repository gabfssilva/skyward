"""Lambda Cloud provider configuration.

Immutable configuration dataclass for Lambda Cloud provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from skyward.api.provider import ProviderConfig

if TYPE_CHECKING:
    from skyward.providers.lambda_cloud.provider import LambdaProvider


@dataclass(frozen=True, slots=True)
class Lambda(ProviderConfig):
    """Lambda Cloud provider configuration.

    Lambda Cloud provides bare-metal GPU instances with Lambda Stack
    pre-installed (NVIDIA drivers, CUDA, ML frameworks). On-demand
    pricing only — no spot instances.

    Parameters
    ----------
    api_key
        API key. Falls back to LAMBDA_API_KEY env var.
    region
        Preferred region (e.g., "us-west-1"). If None, auto-selects
        from regions with capacity.
    request_timeout
        HTTP request timeout in seconds. Default: 30.
    """

    api_key: str | None = None
    region: str | None = None
    request_timeout: int = 30

    async def create_provider(self) -> LambdaProvider:
        from skyward.providers.lambda_cloud.provider import LambdaProvider

        return await LambdaProvider.create(self)

    @property
    def type(self) -> str:
        return "lambda"
