"""Verda Cloud provider for Skyward v2.

Verda (formerly DataCrunch) is a cloud provider specializing in AI and ML services.
Features dedicated GPU instances, GPU clusters, and serverless inference.

NOTE: Only config classes are imported at package level to avoid deps.
For handlers and modules, import explicitly:

    from skyward.providers.verda.handler import VerdaHandler
    from skyward.providers.verda.client import VerdaClient

Environment Variables:
    VERDA_CLIENT_ID: API client ID (required if not passed directly)
    VERDA_CLIENT_SECRET: API client secret (required if not passed directly)
"""

# Only config - no heavy dependencies
from .config import Verda


# Lazy imports for backward compatibility
def __getattr__(name: str):
    if name == "VerdaHandler":
        from .handler import VerdaHandler
        return VerdaHandler
    if name in ("VerdaClient", "VerdaError"):
        from .client import VerdaClient, VerdaError
        if name == "VerdaClient":
            return VerdaClient
        return VerdaError
    if name == "VerdaClusterState":
        from .state import VerdaClusterState
        return VerdaClusterState
    if name == "VerdaModule":
        from injector import Module, provider, singleton
        import httpx

        from .client import VERDA_API_BASE, VerdaAuth, VerdaClient, get_credentials

        class VerdaModule(Module):
            """DI module for Verda provider."""

            @singleton
            @provider
            def provide_verda_auth(self, config: Verda) -> VerdaAuth:
                """Provide Verda OAuth2 authentication."""
                client_id = config.client_id
                client_secret = config.client_secret
                if not client_id or not client_secret:
                    client_id, client_secret = get_credentials()
                return VerdaAuth(client_id, client_secret)

            @singleton
            @provider
            def provide_http_client(self, auth: VerdaAuth) -> httpx.AsyncClient:
                """Provide singleton httpx.AsyncClient for Verda API."""
                return httpx.AsyncClient(
                    base_url=VERDA_API_BASE,
                    auth=auth,
                    timeout=60,
                )

            @singleton
            @provider
            def provide_verda_client(self, http_client: httpx.AsyncClient) -> VerdaClient:
                """Provide Verda API client."""
                return VerdaClient(http_client)

        return VerdaModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Config (always available)
    "Verda",
    # Lazy (loaded on demand)
    "VerdaClient",
    "VerdaError",
    "VerdaHandler",
    "VerdaClusterState",
    "VerdaModule",
]
