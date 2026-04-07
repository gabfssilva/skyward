from __future__ import annotations

import typing
from dataclasses import dataclass

from skyward.api.provider import ProviderConfig

if typing.TYPE_CHECKING:
    from skyward.providers.lambda_cloud.provider import LambdaCloudProvider


@dataclass(frozen=True, slots=True)
class LambdaCloud(ProviderConfig):
    """Lambda Cloud provider configuration.

    Parameters
    ----------
    api_key
        API key. Falls back to ``LAMBDA_API_KEY`` env var.
    region
        Preferred region (e.g., ``"us-east-3"``). ``None`` picks first
        region with available capacity.
    request_timeout
        HTTP request timeout in seconds.
    """

    api_key: str | None = None
    region: str | None = None
    request_timeout: int = 30

    @property
    def type(self) -> str:
        return "lambda"

    async def create_provider(self) -> LambdaCloudProvider:
        from skyward.providers.lambda_cloud.provider import LambdaCloudProvider

        return await LambdaCloudProvider.create(self)
