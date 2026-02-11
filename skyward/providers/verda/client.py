"""Async HTTP client for Verda API."""

from __future__ import annotations

import os
from typing import Any

from skyward.infra.http import HttpClient, HttpError

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


class VerdaClient:
    def __init__(self, http_client: HttpClient) -> None:
        self._http = http_client

    async def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        try:
            return await self._http.request(method, path, json=json, params=params)
        except HttpError as e:
            raise VerdaError(f"API error {e.status}: {e.body}") from e

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
        import json as json_mod

        from loguru import logger

        try:
            text = await self._http.request(
                "POST", "/scripts", json={"name": name, "script": script}, format="text"
            )
            text = text.strip()
            logger.debug(f"Verda create_startup_script response: {text[:200]!r}")

            if not text.startswith("{") and not text.startswith("["):
                script_id = text.strip('"')
                return {"id": script_id, "name": name, "script": script}

            return json_mod.loads(text)
        except HttpError as e:
            raise VerdaError(f"API error {e.status}: {e.body}") from e

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

        import json as json_mod

        from loguru import logger

        try:
            text = await self._http.request("POST", "/instances", json=body, format="text")
            text = text.strip()
            logger.debug(f"Verda create_instance response: {text[:200]!r}")

            if not text.startswith("{") and not text.startswith("["):
                instance_id = text.strip('"')
                return {
                    "id": instance_id,
                    "hostname": hostname or "",
                    "status": "creating",
                    "ip": "",
                    "is_spot": is_spot,
                }

            return json_mod.loads(text)
        except HttpError as e:
            raise VerdaError(f"API error {e.status}: {e.body}") from e

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
        """Delete an instance and permanently delete its volumes.

        Per Verda API docs:
        - volume_ids: array of volume IDs to delete
        - delete_permanently: only works when volume_ids is provided
        """
        # Get instance to retrieve attached volume IDs
        instance = await self.get_instance(instance_id)
        volume_ids = instance.get("volume_ids", []) if instance else []

        body: dict[str, Any] = {"id": instance_id, "action": "delete"}
        if volume_ids:
            body["volume_ids"] = volume_ids
            body["delete_permanently"] = True

        await self._request("PUT", "/instances", json=body)

    # =========================================================================
    # Volume Management
    # =========================================================================

    async def delete_volume(self, volume_id: str) -> None:
        """Permanently delete a volume (cannot be undone)."""
        await self._request("DELETE", f"/volumes/{volume_id}", json={"is_permanent": True})


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
