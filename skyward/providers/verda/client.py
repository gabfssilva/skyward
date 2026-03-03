"""Async HTTP client for Verda API."""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from skyward.infra.http import HttpError
from skyward.observability.logger import logger

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


class _OAuth2(httpx.Auth):
    """OAuth2 client-credentials flow with automatic 401 retry."""

    def __init__(self, client_id: str, client_secret: str, token_url: str) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._token_url = token_url
        self._token: str | None = None

    async def _fetch_token(self) -> str:
        async with httpx.AsyncClient() as http:
            resp = await http.post(
                self._token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                },
            )
            resp.raise_for_status()
            return resp.json()["access_token"]

    async def async_auth_flow(
        self, request: httpx.Request,
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        if self._token is None:
            self._token = await self._fetch_token()

        request.headers["Authorization"] = f"Bearer {self._token}"
        response = yield request

        if response.status_code == 401:
            self._token = await self._fetch_token()
            request.headers["Authorization"] = f"Bearer {self._token}"
            yield request


class VerdaClient:
    def __init__(self, http_client: httpx.AsyncClient) -> None:
        self._http = http_client
        self._log = logger.bind(provider="verda", component="client")

    async def close(self) -> None:
        await self._http.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        self._log.debug("{method} {path}", method=method, path=path)
        resp = await self._http.request(method, path, json=json, params=params)
        if resp.status_code >= 400:
            self._log.warning(
                "API error {method} {path}: {status}",
                method=method, path=path, status=resp.status_code,
            )
            raise VerdaError(f"API error {resp.status_code}: {resp.text}")
        return resp.json() if resp.content else None

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
        self._log.debug("Creating SSH key '{name}'", name=name)
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

        try:
            resp = await self._http.request(
                "POST", "/scripts", json={"name": name, "script": script},
            )
            if resp.status_code >= 400:
                raise HttpError(status=resp.status_code, body=resp.text)
            text = resp.text
            text = text.strip()
            self._log.debug("create_startup_script response: {resp}", resp=text[:200])

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
            "location_code": location,
            "is_spot": is_spot,
        }

        if hostname:
            body["hostname"] = hostname
        if description:
            body["description"] = description
        if startup_script_id:
            body["startup_script_id"] = startup_script_id

        import json as json_mod

        try:
            resp = await self._http.request("POST", "/instances", json=body)
            if resp.status_code >= 400:
                raise HttpError(status=resp.status_code, body=resp.text)
            text = resp.text
            text = text.strip()
            self._log.debug("create_instance response: {resp}", resp=text[:200])

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
            result: InstanceResponse | None = await self._request(
                "GET", f"/instances/{instance_id}",
            )
            return result
        except VerdaError:
            return None

    async def list_instances(self, status: str | None = None) -> list[InstanceResponse]:
        """List all instances, optionally filtered by status."""
        params = {"status": status} if status else None
        result: list[InstanceResponse] | None = await self._request(
            "GET", "/instances", params=params,
        )
        return result or []

    async def delete_instance(self, instance_id: str) -> None:
        """Delete an instance and permanently delete its volumes.

        Per Verda API docs:
        - volume_ids: array of volume IDs to delete
        - delete_permanently: only works when volume_ids is provided
        """
        self._log.debug("Deleting instance {iid}", iid=instance_id)
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
        raise ValueError(
            "Verda client secret not found. "
            "Set VERDA_CLIENT_SECRET environment variable."
        )

    return client_id, client_secret
