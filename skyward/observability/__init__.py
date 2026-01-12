"""Observability modules for Skyward v2.

Provides monitoring and visualization tools including:
- Panel: Rich terminal dashboard with metrics, logs, and cost tracking
"""

from injector import Module, provider, singleton

from skyward.bus import AsyncEventBus
from skyward.spec import PoolSpec

from .panel import PanelComponent


class PanelModule(Module):
    """DI module that enables the Panel UI.

    Add this module to your injector to enable the Rich terminal
    dashboard. The panel automatically subscribes to events and
    displays cluster status, metrics, logs, and cost.

    Usage:
        modules = [
            SkywardModule(),
            PoolConfigModule(spec=spec, provider_config=provider),
            PanelModule(),  # Add this to enable panel
        ]

        injector = Injector(modules)
    """

    @singleton
    @provider
    def provide_panel(self, bus: AsyncEventBus, spec: PoolSpec) -> PanelComponent:
        """Provide the panel component as a singleton."""
        return PanelComponent(bus=bus, spec=spec)


# Logging configuration
from .logging import (
    CONSOLE_FORMAT,
    FILE_FORMAT,
    LogConfig,
    LogLevel,
    _setup_logging,
    _teardown_logging,
)

__all__ = [
    "PanelModule",
    "PanelComponent",
    # Logging
    "LogConfig",
    "LogLevel",
    "CONSOLE_FORMAT",
    "FILE_FORMAT",
    "_setup_logging",
    "_teardown_logging",
]
