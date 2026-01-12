"""AWS Provider for Skyward v2.

Event-driven AWS provisioning with dependency injection.

NOTE: Only config classes are imported at package level to avoid SDK deps.
For handlers and modules, import explicitly:

    from skyward.providers.aws.handler import AWSHandler
    from skyward.providers.aws.clients import AWSModule, Client
"""

# Only config - no SDK dependencies
from .config import AWS, AllocationStrategy

# Lazy imports for backward compatibility (only load when accessed)
def __getattr__(name: str):
    if name in ("AWSHandler",):
        from .handler import AWSHandler
        return AWSHandler
    if name in ("AWSModule", "Client"):
        from .clients import AWSModule, Client
        if name == "AWSModule":
            return AWSModule
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
    "AWSHandler",
    "AWSModule",
    "Client",
    "AWSClusterState",
    "AWSResources",
    "InstanceConfig",
]
