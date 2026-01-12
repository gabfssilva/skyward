"""Central DI module for Skyward v2.

Provides core dependencies that all components need:
- AsyncEventBus (singleton)
- PoolSpec (configured per-pool)
- Provider config (AWS, VastAI, Verda, etc.)
"""

from __future__ import annotations

from typing import Any

from injector import Binder, Module, provider, singleton

from .bus import AsyncEventBus
from .spec import PoolSpec


class SkywardModule(Module):
    """Core module providing shared dependencies.

    Usage:
        injector = Injector([SkywardModule(), AWSModule()])
        pool = injector.get(ComputePool)
    """

    @singleton
    @provider
    def provide_bus(self) -> AsyncEventBus:
        """Provide singleton event bus."""
        return AsyncEventBus()


class PoolConfigModule(Module):
    """Module for pool-specific configuration.

    Binds PoolSpec and provider config for a specific pool instance.
    """

    def __init__(
        self,
        spec: PoolSpec,
        provider_config: Any = None,
    ) -> None:
        self._spec = spec
        self._provider_config = provider_config

    def configure(self, binder: Binder) -> None:
        """Configure bindings for pool-specific dependencies."""
        from .providers.aws import AWS
        from .providers.vastai import VastAI
        from .providers.verda import Verda

        # Bind PoolSpec
        binder.bind(PoolSpec, to=self._spec)

        # Bind provider config based on type
        if self._provider_config is None:
            # Default to AWS
            binder.bind(AWS, to=AWS())
        elif isinstance(self._provider_config, AWS):
            binder.bind(AWS, to=self._provider_config)
        elif isinstance(self._provider_config, VastAI):
            binder.bind(VastAI, to=self._provider_config)
        elif isinstance(self._provider_config, Verda):
            binder.bind(Verda, to=self._provider_config)


__all__ = [
    "SkywardModule",
    "PoolConfigModule",
]
