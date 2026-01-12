"""Provider registry for automatic discovery.

Provides a registration mechanism for providers that eliminates
if/elif chains when selecting providers based on configuration.

Uses lazy loading to avoid importing heavy SDK dependencies until needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from injector import Module


def get_provider_for_config(config: Any) -> tuple[type, type["Module"], str]:
    """Get provider classes for a configuration object.

    Uses lazy imports to only load SDK dependencies when needed.

    Args:
        config: A provider configuration instance.

    Returns:
        Tuple of (HandlerClass, ModuleClass, provider_name).

    Raises:
        ValueError: If no provider is registered for the config type.
    """
    # Import config classes for type checking (lightweight)
    from .aws.config import AWS
    from .vastai.config import VastAI
    from .verda.config import Verda

    # Lazy load handlers and modules based on config type
    if isinstance(config, AWS):
        from .aws.handler import AWSHandler
        from .aws.clients import AWSModule

        return AWSHandler, AWSModule, "aws"

    if isinstance(config, VastAI):
        from .vastai.handler import VastAIHandler
        from .vastai import VastAIModule

        return VastAIHandler, VastAIModule, "vastai"

    if isinstance(config, Verda):
        from .verda.handler import VerdaHandler
        from .verda import VerdaModule

        return VerdaHandler, VerdaModule, "verda"

    raise ValueError(
        f"No provider registered for {type(config).__name__}. "
        f"Available providers: AWS, VastAI, Verda"
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "get_provider_for_config",
]
