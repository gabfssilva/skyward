"""Provider registry for automatic discovery.

Provides a registration mechanism for providers that eliminates
if/elif chains when selecting providers based on configuration.

Uses lazy loading to avoid importing heavy SDK dependencies until needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from casty import Behavior

from skyward.actors.messages import ProviderMsg
from skyward.observability.logger import logger

if TYPE_CHECKING:
    from .aws.config import AWS
    from .runpod.config import RunPod
    from .vastai.config import VastAI
    from .verda.config import Verda

log = logger.bind(component="registry")

type ProviderConfig = AWS | RunPod | VastAI | Verda
type ProviderActorFactory = Any


async def create_provider(config: ProviderConfig) -> Any:
    """Create a stateless CloudProvider for a configuration object.

    Uses lazy imports to only load SDK dependencies when needed.

    Returns:
        A CloudProvider instance ready for prepare/provision/terminate/teardown.
    """
    from .aws.config import AWS
    from .runpod.config import RunPod
    from .vastai.config import VastAI
    from .verda.config import Verda

    config_type = type(config).__name__
    log.debug("Creating provider for config={config_type}", config_type=config_type)

    match config:
        case AWS():
            from .aws.provider import AWSCloudProvider
            return await AWSCloudProvider.create(config)
        case VastAI():
            from .vastai.provider import VastAICloudProvider
            return await VastAICloudProvider.create(config)
        case Verda():
            from .verda.provider import VerdaCloudProvider
            return await VerdaCloudProvider.create(config)
        case RunPod():
            from .runpod.provider import RunPodCloudProvider
            return await RunPodCloudProvider.create(config)
        case _:
            raise ValueError(
                f"No provider registered for {type(config).__name__}. "
                f"Available providers: AWS, VastAI, Verda, RunPod"
            )


def get_provider_for_config(config: ProviderConfig) -> tuple[ProviderActorFactory, str]:
    """Legacy: Get provider actor factory for a configuration object.

    Kept for backward compatibility with old handler-based approach.
    New code should use create_provider() instead.
    """
    from .aws.config import AWS
    from .runpod.config import RunPod
    from .vastai.config import VastAI
    from .verda.config import Verda

    config_type = type(config).__name__
    log.debug("Resolving provider for config={config_type}", config_type=config_type)

    match config:
        case AWS():
            from .aws.clients import EC2ClientFactory

            def aws_factory(cfg: ProviderConfig) -> Behavior[ProviderMsg]:
                from collections.abc import AsyncIterator
                from contextlib import asynccontextmanager

                import aioboto3  # type: ignore[reportMissingImports]

                from .aws.handler import aws_provider_actor

                aws_cfg: AWS = cfg  # type: ignore[assignment]

                @asynccontextmanager
                async def ec2_factory() -> AsyncIterator[Any]:
                    session = aioboto3.Session(region_name=aws_cfg.region)
                    async with session.client("ec2", region_name=aws_cfg.region) as ec2:  # type: ignore[reportGeneralTypeIssues]
                        yield ec2

                return aws_provider_actor(aws_cfg, EC2ClientFactory(ec2_factory))

            return aws_factory, "aws"

        case VastAI():
            def vastai_factory(cfg: ProviderConfig) -> Behavior[ProviderMsg]:
                from .vastai.client import VastAIClient, get_api_key
                from .vastai.handler import vastai_provider_actor

                vastai_cfg: VastAI = cfg  # type: ignore[assignment]
                api_key = vastai_cfg.api_key or get_api_key()
                client = VastAIClient(api_key, config=vastai_cfg)
                return vastai_provider_actor(vastai_cfg, client)

            return vastai_factory, "vastai"

        case Verda():
            def verda_factory(cfg: ProviderConfig) -> Behavior[ProviderMsg]:
                from skyward.infra.http import HttpClient, OAuth2Auth

                from .verda.client import VERDA_API_BASE, VerdaClient, get_credentials
                from .verda.handler import verda_provider_actor

                verda_cfg: Verda = cfg  # type: ignore[assignment]
                client_id = verda_cfg.client_id
                client_secret = verda_cfg.client_secret
                if not client_id or not client_secret:
                    client_id, client_secret = get_credentials()
                auth = OAuth2Auth(client_id, client_secret, f"{VERDA_API_BASE}/oauth2/token")
                http_client = HttpClient(VERDA_API_BASE, auth, timeout=verda_cfg.request_timeout)
                client = VerdaClient(http_client)
                return verda_provider_actor(verda_cfg, client)

            return verda_factory, "verda"

        case RunPod():
            def runpod_factory(cfg: ProviderConfig) -> Behavior[ProviderMsg]:
                from .runpod.handler import runpod_provider_actor

                runpod_cfg: RunPod = cfg  # type: ignore[assignment]
                return runpod_provider_actor(runpod_cfg)

            return runpod_factory, "runpod"

        case _:
            raise ValueError(
                f"No provider registered for {type(config).__name__}. "
                f"Available providers: AWS, VastAI, Verda, RunPod"
            )
