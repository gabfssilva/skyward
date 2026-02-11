"""Provider registry for automatic discovery.

Provides a registration mechanism for providers that eliminates
if/elif chains when selecting providers based on configuration.

Uses lazy loading to avoid importing heavy SDK dependencies until needed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from casty import ActorRef, Behavior
from loguru import logger

from skyward.actors.messages import ProviderMsg

log = logger.bind(component="registry")

type ProviderActorFactory = Callable[[Any, ActorRef], Behavior[ProviderMsg]]


def get_provider_for_config(config: Any) -> tuple[ProviderActorFactory, str]:
    """Get provider actor factory for a configuration object.

    Uses lazy imports to only load SDK dependencies when needed.

    Returns:
        Tuple of (actor_factory, provider_name).
        actor_factory(config, pool_ref) -> Behavior[ProviderMsg]
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

            def aws_factory(cfg: Any, pool_ref: ActorRef) -> Behavior[ProviderMsg]:
                from collections.abc import AsyncIterator
                from contextlib import asynccontextmanager

                import aioboto3  # type: ignore[reportMissingImports]

                from .aws.handler import aws_provider_actor

                @asynccontextmanager
                async def ec2_factory() -> AsyncIterator[Any]:
                    session = aioboto3.Session(region_name=cfg.region)
                    async with session.client("ec2", region_name=cfg.region) as ec2:  # type: ignore[reportGeneralTypeIssues]
                        yield ec2

                return aws_provider_actor(cfg, EC2ClientFactory(ec2_factory), pool_ref)

            return aws_factory, "aws"

        case VastAI():
            def vastai_factory(cfg: Any, pool_ref: ActorRef) -> Behavior[ProviderMsg]:
                from .vastai.client import VastAIClient, get_api_key
                from .vastai.handler import vastai_provider_actor
                api_key = cfg.api_key or get_api_key()
                client = VastAIClient(api_key, config=cfg)
                return vastai_provider_actor(cfg, client, pool_ref)

            return vastai_factory, "vastai"

        case Verda():
            def verda_factory(cfg: Any, pool_ref: ActorRef) -> Behavior[ProviderMsg]:
                from skyward.infra.http import HttpClient, OAuth2Auth

                from .verda.client import VERDA_API_BASE, VerdaClient, get_credentials
                from .verda.handler import verda_provider_actor

                client_id = cfg.client_id
                client_secret = cfg.client_secret
                if not client_id or not client_secret:
                    client_id, client_secret = get_credentials()
                auth = OAuth2Auth(client_id, client_secret, f"{VERDA_API_BASE}/oauth2/token")
                http_client = HttpClient(VERDA_API_BASE, auth, timeout=60)
                client = VerdaClient(http_client)
                return verda_provider_actor(cfg, client, pool_ref)

            return verda_factory, "verda"

        case RunPod():
            def runpod_factory(cfg: Any, pool_ref: ActorRef) -> Behavior[ProviderMsg]:
                from .runpod.handler import runpod_provider_actor
                return runpod_provider_actor(cfg, pool_ref)

            return runpod_factory, "runpod"

        case _:
            raise ValueError(
                f"No provider registered for {type(config).__name__}. "
                f"Available providers: AWS, VastAI, Verda, RunPod"
            )
