"""RunPod GPU Pods provider for Skyward.

RunPod provides GPU cloud computing with both Secure Cloud (dedicated)
and Community Cloud (marketplace) options.

NOTE: Only config classes are imported at package level to avoid deps.
For handlers and modules, import explicitly:

    from skyward.providers.runpod.handler import RunPodHandler
    from skyward.providers.runpod.client import RunPodClient

Environment Variables:
    RUNPOD_API_KEY: API key (required if not passed directly)
"""

# Only config - no heavy dependencies
from .config import CloudType, RunPod


# Lazy imports for backward compatibility
def __getattr__(name: str):
    if name == "RunPodHandler":
        from .handler import RunPodHandler

        return RunPodHandler
    if name in ("RunPodClient", "RunPodError"):
        from .client import RunPodClient, RunPodError

        if name == "RunPodClient":
            return RunPodClient
        return RunPodError
    if name == "RunPodClusterState":
        from .state import RunPodClusterState

        return RunPodClusterState
    if name == "RunPodModule":
        from injector import Module, provider, singleton

        from .client import RunPodClient, get_api_key

        class RunPodModule(Module):
            """DI module for RunPod provider."""

            @singleton
            @provider
            def provide_runpod_client(self, config: RunPod) -> RunPodClient:
                """Provide RunPod API client."""
                api_key = get_api_key(config.api_key)
                return RunPodClient(api_key)

        return RunPodModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Config (always available)
    "CloudType",
    "RunPod",
    # Lazy (loaded on demand)
    "RunPodClient",
    "RunPodError",
    "RunPodHandler",
    "RunPodClusterState",
    "RunPodModule",
]
