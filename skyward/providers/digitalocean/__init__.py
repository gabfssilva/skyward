"""DigitalOcean provider for Skyward v2.

DigitalOcean provides GPU droplets with H100 GPUs.
No spot pricing available.

Example:
    from skyward import ComputePool, PoolSpec, app_context
    from skyward.providers.digitalocean import DOModule, DigitalOcean

    async def main():
        spec = PoolSpec(nodes=1, accelerator="H100", allocation="on-demand")
        async with app_context(DOModule()) as app:
            pool = app.get(ComputePool)
            async with pool:
                print(f"Cluster ready: {pool.cluster_id}")

Environment Variables:
    DIGITALOCEAN_TOKEN: API token (required if not passed directly)
    DIGITALOCEAN_SSH_KEY_FINGERPRINT: SSH key fingerprint (optional)
"""

from injector import Module, provider, singleton

from .client import DigitalOceanClient, DigitalOceanError, get_token
from .config import DigitalOcean
from .handler import DOHandler
from .state import DOClusterState
from .types import DropletResponse, SizeResponse, SSHKeyResponse


class DOModule(Module):
    """DI module for DigitalOcean provider."""

    @singleton
    @provider
    def provide_config(self) -> DigitalOcean:
        """Provide default DigitalOcean configuration."""
        return DigitalOcean()


__all__ = [
    # Config
    "DigitalOcean",
    # Client
    "DigitalOceanClient",
    "DigitalOceanError",
    "get_token",
    # Handler
    "DOHandler",
    # State
    "DOClusterState",
    # Types
    "DropletResponse",
    "SizeResponse",
    "SSHKeyResponse",
    # Module
    "DOModule",
]
