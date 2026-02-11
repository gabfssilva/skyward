"""AWS Provider for Skyward v2.

Event-driven AWS provisioning with dependency injection.

NOTE: Only config classes are imported at package level to avoid SDK deps.
For handlers and modules, import explicitly:

    from skyward.providers.aws.handler import AWSHandler
    from skyward.providers.aws.clients import Client
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .clients import Client
    from .handler import aws_provider_actor
    from .state import AWSClusterState, AWSResources, InstanceConfig

# Only config - no SDK dependencies
from .config import AWS, AllocationStrategy


# Lazy imports for backward compatibility (only load when accessed)
def __getattr__(name: str) -> Any:
    if name == "aws_provider_actor":
        from .handler import aws_provider_actor
        return aws_provider_actor
    if name == "Client":
        from .clients import Client
        return Client
    if name in ("AWSClusterState", "AWSResources", "InstanceConfig"):
        from .state import AWSClusterState, AWSResources, InstanceConfig
        if name == "AWSClusterState":
            return AWSClusterState
        if name == "AWSResources":
            return AWSResources
        return InstanceConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Config (always available)
    "AWS",
    "AllocationStrategy",
    # Lazy (loaded on demand)
    "aws_provider_actor",
    "Client",
    "AWSClusterState",
    "AWSResources",
    "InstanceConfig",
]
