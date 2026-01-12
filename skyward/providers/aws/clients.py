"""AWS client factories with dependency injection.

Provides typed client factories that can be injected into components.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING, Any

import aioboto3
from injector import Module, provider, singleton

from .config import AWS

if TYPE_CHECKING:
    from types_aiobotocore_ec2 import EC2Client


# =============================================================================
# Client Type
# =============================================================================

type Client[T] = Callable[[], AbstractAsyncContextManager[T]]
"""Factory that returns an async context manager for a client."""

# =============================================================================
# Wrapper Classes for DI (each needs a unique type)
# =============================================================================


class EC2ClientFactory:
    """Wrapper for EC2 client factory."""

    def __init__(self, factory: Callable[[], AbstractAsyncContextManager[Any]]) -> None:
        self._factory = factory

    def __call__(self) -> AbstractAsyncContextManager[Any]:
        return self._factory()


# =============================================================================
# AWS Module
# =============================================================================


class AWSModule(Module):
    """DI module that provides AWS client factories.

    Usage:
        >>> from injector import Injector
        >>> from skyward.providers.aws import AWSModule, AWS
        >>>
        >>> injector = Injector([AWSModule()])
        >>> injector.binder.bind(AWS, to=AWS(region="us-east-1"))
        >>>
        >>> # In a component:
        >>> class MyHandler:
        ...     ec2: Client[EC2Client]
        ...
        ...     async def do_something(self):
        ...         async with self.ec2() as client:
        ...             await client.describe_instances()
    """

    @singleton
    @provider
    def provide_session(self) -> aioboto3.Session:
        """Provide singleton aioboto3 session."""
        return aioboto3.Session()

    @singleton
    @provider
    def provide_ec2(self, session: aioboto3.Session, config: AWS) -> EC2ClientFactory:
        """Provide EC2 client factory."""
        @asynccontextmanager
        async def factory() -> AsyncIterator[Any]:
            async with session.client("ec2", region_name=config.region) as client:
                yield client
        return EC2ClientFactory(factory)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "Client",
    "AWSModule",
]
