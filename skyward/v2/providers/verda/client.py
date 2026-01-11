"""Async HTTP client for Verda API.

Uses httpx for async HTTP requests with OAuth2 authentication.
Returns TypedDicts directly.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from skyward.v2.app import component

from .config import Verda
from .types import (
    AvailabilityRegion,
    InstanceResponse,
    InstanceTypeResponse,
    SSHKeyResponse,
    StartupScriptResponse,
)

VERDA_API_BASE = "https://api.verda.com/v1"


class VerdaError(Exception):
    """Error from Verda API."""


@component
class VerdaClient:
    """Async HTTP client for Verda API.

    Returns TypedDicts directly from API responses.
    Config is injected via DI.

    Example:
        # Via DI injection in a handler:
        class MyHandler:
            client: VerdaClient

            async def do_something(self):
                async with self.client:
                    types = await self.client.list_instance_types()
    """

    config: Verda

    def __post_init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._access_token: str | None = None

    async def __aenter__(self) -> VerdaClient:
        client_id = self.config.client_id
        client_secret = self.config.client_secret
        if not client_id or not client_secret:
            client_id, client_secret = get_credentials()

        self._client = httpx.AsyncClient(
            base_url=VERDA_API_BASE,
            timeout=60,
        )
        await self._authenticate(client_id, client_secret)
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if not self._client:
            raise RuntimeError("Client not initialized. Use 'async with' context.")
        return self._client

    async def _authenticate(self, client_id: str, client_secret: str) -> None:
        """Authenticate with OAuth2 client credentials."""
        resp = await self.client.post(
            "/oauth2/token",
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
        )
        resp.raise_for_status()
        self._access_token = resp.json()["access_token"]

    def _headers(self) -> dict[str, str]:
        if not self._access_token:
            raise RuntimeError("Not authenticated")
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Accept": "application/json",
        }

    async def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Execute HTTP request and return JSON response."""
        try:
            resp = await self.client.request(
                method, path, headers=self._headers(), json=json, params=params
            )
            resp.raise_for_status()
            return resp.json() if resp.content else None
        except httpx.HTTPStatusError as e:
            raise VerdaError(f"API error {e.response.status_code}: {e.response.text}") from e
        except httpx.RequestError as e:
            raise VerdaError(f"Request failed: {e}") from e

    # =========================================================================
    # Instance Types
    # =========================================================================

    async def list_instance_types(self) -> list[InstanceTypeResponse]:
        """List available instance types with pricing."""
        result: list[InstanceTypeResponse] | None = await self._request("GET", "/instance-types")
        return result or []

    async def get_availability(self, is_spot: bool = False) -> dict[str, frozenset[str]]:
        """Get instance availability across all regions.

        Returns:
            Dict mapping region code to set of available instance types.
        """
        result: list[AvailabilityRegion] | None = await self._request(
            "GET", "/instance-availability", params={"is_spot": str(is_spot).lower()}
        )
        if not result:
            return {}
        return {
            region["location_code"]: frozenset(region.get("availabilities", []))
            for region in result
        }

    # =========================================================================
    # SSH Key Management
    # =========================================================================

    async def list_ssh_keys(self) -> list[SSHKeyResponse]:
        """List all SSH keys registered on this account."""
        result: list[SSHKeyResponse] | None = await self._request("GET", "/ssh-keys")
        return result or []

    async def create_ssh_key(self, name: str, public_key: str) -> SSHKeyResponse:
        """Register a new SSH public key."""
        result: SSHKeyResponse | None = await self._request(
            "POST", "/ssh-keys", json={"name": name, "key": public_key}
        )
        if not result:
            raise VerdaError("Failed to create SSH key: empty response")
        return result

    async def delete_ssh_key(self, key_id: str) -> None:
        """Delete an SSH key."""
        await self._request("DELETE", f"/ssh-keys/{key_id}")

    # =========================================================================
    # Startup Scripts
    # =========================================================================

    async def list_startup_scripts(self) -> list[StartupScriptResponse]:
        """List all startup scripts."""
        result: list[StartupScriptResponse] | None = await self._request("GET", "/scripts")
        return result or []

    async def create_startup_script(self, name: str, script: str) -> StartupScriptResponse:
        """Create a new startup script."""
        from loguru import logger
        try:
            resp = await self.client.request(
                "POST", "/scripts", headers=self._headers(), json={"name": name, "script": script}
            )
            resp.raise_for_status()

            # Handle different response formats
            if not resp.content:
                raise VerdaError("Failed to create startup script: empty response")

            text = resp.text.strip()
            logger.debug(f"Verda create_startup_script response: {text[:200]!r}")

            # API returns just the ID (UUID, number, or quoted string)
            # Check if it looks like a plain ID (not JSON object/array)
            if not text.startswith('{') and not text.startswith('['):
                script_id = text.strip('"')
                return {"id": script_id, "name": name, "script": script}

            # Try parsing as JSON
            import json
            return json.loads(text)
        except httpx.HTTPStatusError as e:
            raise VerdaError(f"API error {e.response.status_code}: {e.response.text}") from e
        except httpx.RequestError as e:
            raise VerdaError(f"Request failed: {e}") from e

    async def delete_startup_script(self, script_id: str) -> None:
        """Delete a startup script."""
        await self._request("DELETE", f"/scripts/{script_id}")

    # =========================================================================
    # Instance Management
    # =========================================================================

    async def create_instance(
        self,
        instance_type: str,
        image: str,
        ssh_key_ids: list[str],
        location: str,
        hostname: str | None = None,
        description: str | None = None,
        startup_script_id: str | None = None,
        is_spot: bool = False,
    ) -> InstanceResponse:
        """Create a new instance."""
        body: dict[str, Any] = {
            "instance_type": instance_type,
            "image": image,
            "ssh_key_ids": ssh_key_ids,
            "location": location,
            "is_spot": is_spot,
        }

        if hostname:
            body["hostname"] = hostname
        if description:
            body["description"] = description
        if startup_script_id:
            body["startup_script_id"] = startup_script_id

        from loguru import logger
        try:
            resp = await self.client.request(
                "POST", "/instances", headers=self._headers(), json=body
            )
            resp.raise_for_status()

            if not resp.content:
                raise VerdaError("Failed to create instance: empty response")

            text = resp.text.strip()
            logger.debug(f"Verda create_instance response: {text[:200]!r}")

            # API returns just the ID (UUID), not a JSON object
            if not text.startswith('{') and not text.startswith('['):
                instance_id = text.strip('"')
                # Return minimal instance info - caller should poll for full details
                return {
                    "id": instance_id,
                    "hostname": hostname or "",
                    "status": "creating",
                    "ip": "",
                    "is_spot": is_spot,
                }

            import json
            return json.loads(text)
        except httpx.HTTPStatusError as e:
            raise VerdaError(f"API error {e.response.status_code}: {e.response.text}") from e
        except httpx.RequestError as e:
            raise VerdaError(f"Request failed: {e}") from e

    async def get_instance(self, instance_id: str) -> InstanceResponse | None:
        """Get instance details. Returns None if not found."""
        try:
            result: InstanceResponse | None = await self._request("GET", f"/instances/{instance_id}")
            return result
        except VerdaError:
            return None

    async def list_instances(self, status: str | None = None) -> list[InstanceResponse]:
        """List all instances, optionally filtered by status."""
        params = {"status": status} if status else None
        result: list[InstanceResponse] | None = await self._request("GET", "/instances", params=params)
        return result or []

    async def delete_instance(self, instance_id: str) -> None:
        """Delete an instance."""
        await self._request("PUT", "/instances", json={"id": instance_id, "action": "delete"})


# =============================================================================
# Utility Functions
# =============================================================================


def get_credentials() -> tuple[str, str]:
    """Get Verda credentials from environment.

    Returns:
        Tuple of (client_id, client_secret).
    """
    client_id = os.environ.get("VERDA_CLIENT_ID")
    client_secret = os.environ.get("VERDA_CLIENT_SECRET")

    if not client_id:
        raise ValueError("Verda client ID not found. Set VERDA_CLIENT_ID environment variable.")
    if not client_secret:
        raise ValueError("Verda client secret not found. Set VERDA_CLIENT_SECRET environment variable.")

    return client_id, client_secret


__all__ = [
    "VerdaClient",
    "VerdaError",
    "get_credentials",
]
