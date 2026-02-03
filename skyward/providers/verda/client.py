"""Async HTTP client for Verda API.

Uses httpx for async HTTP requests with OAuth2 authentication.
Returns TypedDicts directly.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator
from typing import Any

import httpx

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


class VerdaAuth(httpx.Auth):
    """OAuth2 client credentials authentication for Verda API.

    Thread-safe and handles token refresh automatically.
    Designed to be used with a singleton httpx.AsyncClient.
    """

    def __init__(self, client_id: str, client_secret: str) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._token: str | None = None
        self._lock = asyncio.Lock()

    async def _fetch_token(self, client: httpx.AsyncClient) -> str:
        """Fetch a new access token."""
        resp = await client.post(
            f"{VERDA_API_BASE}/oauth2/token",
            data={
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
            },
        )
        resp.raise_for_status()
        return resp.json()["access_token"]

    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        """Add authorization header to request, fetching token if needed."""
        async with self._lock:
            if not self._token:
                # Create a temporary client for token fetch (no auth to avoid recursion)
                async with httpx.AsyncClient(timeout=30) as token_client:
                    self._token = await self._fetch_token(token_client)

        request.headers["Authorization"] = f"Bearer {self._token}"
        request.headers["Accept"] = "application/json"

        response = yield request

        # Handle token expiration (401 Unauthorized)
        if response.status_code == 401:
            async with self._lock:
                async with httpx.AsyncClient(timeout=30) as token_client:
                    self._token = await self._fetch_token(token_client)

            request.headers["Authorization"] = f"Bearer {self._token}"
            yield request


class VerdaClient:
    """Async HTTP client for Verda API.

    Returns TypedDicts directly from API responses.
    httpx.AsyncClient is injected via DI as a singleton.

    Example:
        # Via DI injection in a handler:
        class MyHandler:
            client: VerdaClient

            async def do_something(self):
                types = await self.client.list_instance_types()
    """

    def __init__(self, http_client: httpx.AsyncClient) -> None:
        self._client = http_client

    async def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Execute HTTP request and return JSON response."""
        try:
            resp = await self._client.request(method, path, json=json, params=params)
            resp.raise_for_status()
            return resp.json() if resp.content else None
        except httpx.HTTPStatusError as e:
            raise VerdaError(f"API error {e.response.status_code}: {e.response.text}") from e
        except httpx.RequestError as e:
            raise VerdaError(f"Request failed ({type(e).__name__}): {e}") from e

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
            resp = await self._client.request(
                "POST", "/scripts", json={"name": name, "script": script}
            )
            resp.raise_for_status()

            if not resp.content:
                raise VerdaError("Failed to create startup script: empty response")

            text = resp.text.strip()
            logger.debug(f"Verda create_startup_script response: {text[:200]!r}")

            # API returns just the ID (UUID, number, or quoted string)
            if not text.startswith("{") and not text.startswith("["):
                script_id = text.strip('"')
                return {"id": script_id, "name": name, "script": script}

            import json

            return json.loads(text)
        except httpx.HTTPStatusError as e:
            raise VerdaError(f"API error {e.response.status_code}: {e.response.text}") from e
        except httpx.RequestError as e:
            raise VerdaError(f"Request failed ({type(e).__name__}): {e}") from e

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
            resp = await self._client.request("POST", "/instances", json=body)
            resp.raise_for_status()

            if not resp.content:
                raise VerdaError("Failed to create instance: empty response")

            text = resp.text.strip()
            logger.debug(f"Verda create_instance response: {text[:200]!r}")

            # API returns just the ID (UUID), not a JSON object
            if not text.startswith("{") and not text.startswith("["):
                instance_id = text.strip('"')
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
            raise VerdaError(f"Request failed ({type(e).__name__}): {e}") from e

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
    "VERDA_API_BASE",
    "VerdaAuth",
    "VerdaClient",
    "VerdaError",
    "get_credentials",
]
