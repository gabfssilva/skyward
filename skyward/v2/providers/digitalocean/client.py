"""Async client for DigitalOcean API.

Uses pydo.aio for async operations. Returns TypedDicts directly.
"""

from __future__ import annotations

import os
from typing import Any, cast

from pydo.aio import Client as PyDOClient

from skyward.v2.app import component

from .config import DigitalOcean
from .types import (
    DropletResponse,
    SizeResponse,
    SSHKeyResponse,
)


class DigitalOceanError(Exception):
    """Error from DigitalOcean API."""


@component
class DigitalOceanClient:
    """Async client for DigitalOcean API using pydo.aio.

    Returns TypedDicts directly from API responses.
    Config is injected via DI.

    Example:
        # Via DI injection in a handler:
        class MyHandler:
            client: DigitalOceanClient

            async def do_something(self):
                async with self.client:
                    sizes = await self.client.list_sizes()
    """

    config: DigitalOcean

    def __post_init__(self) -> None:
        self._client: PyDOClient | None = None

    async def __aenter__(self) -> DigitalOceanClient:
        token = self.config.token or get_token()
        self._client = PyDOClient(token=token)
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    @property
    def client(self) -> PyDOClient:
        if not self._client:
            raise RuntimeError("Client not initialized. Use 'async with' context.")
        return self._client

    # =========================================================================
    # SSH Key Management
    # =========================================================================

    async def list_ssh_keys(self) -> list[SSHKeyResponse]:
        """List all SSH keys registered on this account."""
        try:
            result = await self.client.ssh_keys.list()
            return cast(list[SSHKeyResponse], result.get("ssh_keys", []))
        except Exception as e:
            raise DigitalOceanError(f"Failed to list SSH keys: {e}") from e

    async def create_ssh_key(self, name: str, public_key: str) -> SSHKeyResponse:
        """Register a new SSH public key."""
        try:
            result = await self.client.ssh_keys.create(
                body={"name": name, "public_key": public_key}
            )
            ssh_key = result.get("ssh_key")
            if not ssh_key:
                raise DigitalOceanError("Failed to create SSH key: empty response")
            return cast(SSHKeyResponse, ssh_key)
        except DigitalOceanError:
            raise
        except Exception as e:
            raise DigitalOceanError(f"Failed to create SSH key: {e}") from e

    async def delete_ssh_key(self, key_id: int) -> None:
        """Delete an SSH key by ID."""
        try:
            await self.client.ssh_keys.delete(ssh_key_identifier=str(key_id))
        except Exception as e:
            raise DigitalOceanError(f"Failed to delete SSH key: {e}") from e

    # =========================================================================
    # Sizes (Instance Types)
    # =========================================================================

    async def list_sizes(self) -> list[SizeResponse]:
        """List all available droplet sizes."""
        sizes: list[SizeResponse] = []
        page = 1

        try:
            while True:
                result = await self.client.sizes.list(page=page, per_page=100)
                page_sizes = result.get("sizes", [])
                sizes.extend(cast(list[SizeResponse], page_sizes))

                # Check for more pages
                if len(page_sizes) < 100:
                    break
                page += 1
        except Exception as e:
            raise DigitalOceanError(f"Failed to list sizes: {e}") from e

        return sizes

    # =========================================================================
    # Droplet Management
    # =========================================================================

    async def create_droplet(
        self,
        name: str,
        region: str,
        size: str,
        image: str,
        ssh_keys: list[str | int],
        user_data: str | None = None,
        tags: list[str] | None = None,
        vpc_uuid: str | None = None,
    ) -> DropletResponse:
        """Create a new droplet."""
        body: dict[str, Any] = {
            "name": name,
            "region": region,
            "size": size,
            "image": image,
            "ssh_keys": ssh_keys,
        }

        if user_data:
            body["user_data"] = user_data
        if tags:
            body["tags"] = tags
        if vpc_uuid:
            body["vpc_uuid"] = vpc_uuid

        try:
            result = await self.client.droplets.create(body=body)
            droplet = result.get("droplet")
            if not droplet:
                raise DigitalOceanError("Failed to create droplet: empty response")
            return cast(DropletResponse, droplet)
        except DigitalOceanError:
            raise
        except Exception as e:
            raise DigitalOceanError(f"Failed to create droplet: {e}") from e

    async def get_droplet(self, droplet_id: int) -> DropletResponse | None:
        """Get droplet details. Returns None if not found."""
        try:
            result = await self.client.droplets.get(droplet_id=droplet_id)
            droplet = result.get("droplet")
            return cast(DropletResponse, droplet) if droplet else None
        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower():
                return None
            raise DigitalOceanError(f"Failed to get droplet: {e}") from e

    async def list_droplets(self, tag_name: str | None = None) -> list[DropletResponse]:
        """List all droplets, optionally filtered by tag."""
        try:
            if tag_name:
                result = await self.client.droplets.list(tag_name=tag_name)
            else:
                result = await self.client.droplets.list()
            return cast(list[DropletResponse], result.get("droplets", []))
        except Exception as e:
            raise DigitalOceanError(f"Failed to list droplets: {e}") from e

    async def delete_droplet(self, droplet_id: int) -> None:
        """Delete a droplet."""
        try:
            await self.client.droplets.destroy(droplet_id=droplet_id)
        except Exception as e:
            raise DigitalOceanError(f"Failed to delete droplet: {e}") from e


# =============================================================================
# Utility Functions
# =============================================================================


def get_token() -> str:
    """Get DigitalOcean API token from environment."""
    token = os.environ.get("DIGITALOCEAN_TOKEN")
    if not token:
        raise ValueError(
            "DigitalOcean API token not found. "
            "Set DIGITALOCEAN_TOKEN environment variable."
        )
    return token


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "DigitalOceanClient",
    "DigitalOceanError",
    "get_token",
]
