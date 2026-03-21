"""Async HTTP client for Scaleway Instance and IAM APIs."""

from __future__ import annotations

import os
from typing import Any

import httpx

from skyward.infra.http import HttpError
from skyward.infra.retry import on_status_code, retry
from skyward.infra.throttle import throttle
from skyward.observability.logger import logger

from .config import Scaleway
from .types import (
    ImageResponse,
    ServerResponse,
    ServerTypeResponse,
    SSHKeyResponse,
)

SCALEWAY_API_BASE = "https://api.scaleway.com"

log = logger.bind(provider="scaleway")


class ScalewayError(Exception):
    """Error from Scaleway API."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class ScalewayClient:
    """Async HTTP client for Scaleway APIs.

    Covers the Instance API (zone-scoped) and the IAM API (global).
    """

    def __init__(self, secret_key: str, zone: str, *, config: Scaleway | None = None) -> None:
        self._zone = zone
        self._config = config or Scaleway()
        self._http = httpx.AsyncClient(
            base_url=SCALEWAY_API_BASE,
            headers={
                "X-Auth-Token": secret_key,
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(self._config.request_timeout),
        )
        self._log = logger.bind(provider="scaleway", component="client", zone=zone)

    async def __aenter__(self) -> ScalewayClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    async def close(self) -> None:
        await self._http.aclose()

    @retry(on=on_status_code(429, 503), max_attempts=5, base_delay=1.0)
    @throttle(max_concurrent=5, interval=0.2)
    async def _do_request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        resp = await self._http.request(method, path, json=json, params=params)
        if resp.status_code >= 400:
            raise HttpError(status=resp.status_code, body=resp.text)
        return resp.json() if resp.content else None

    async def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        self._log.debug("{method} {path}", method=method, path=path)
        try:
            return await self._do_request(method, path, json=json, params=params)
        except HttpError as e:
            self._log.warning(
                "API error {method} {path}: {status}",
                method=method, path=path, status=e.status,
            )
            raise ScalewayError(f"API error {e.status}: {e.body}", status=e.status) from e

    def _instance_path(self, path: str) -> str:
        """Build zone-scoped Instance API path."""
        return f"/instance/v1/zones/{self._zone}{path}"

    def _block_path(self, path: str) -> str:
        """Build zone-scoped Block Storage API path."""
        return f"/block/v1alpha1/zones/{self._zone}{path}"

    # =========================================================================
    # Server types (discovery)
    # =========================================================================

    async def list_server_types(self) -> dict[str, ServerTypeResponse]:
        """List available server types in the zone.

        Returns a dict mapping commercial_type name to its spec.
        """
        result = await self._request("GET", self._instance_path("/products/servers"))
        return result.get("servers", {}) if result else {}

    # =========================================================================
    # Servers
    # =========================================================================

    async def create_server(
        self,
        name: str,
        commercial_type: str,
        image: str,
        project: str,
        *,
        tags: list[str] | None = None,
    ) -> ServerResponse:
        """Create a server.

        The server is created in stopped state. Call power_on() after.
        """
        payload: dict[str, Any] = {
            "name": name,
            "commercial_type": commercial_type,
            "image": image,
            "project": project,
            "dynamic_ip_required": True,
            "volumes": {},
            "tags": tags or ["skyward"],
        }
        result = await self._request("POST", self._instance_path("/servers"), json=payload)
        return result["server"]

    async def get_server(self, server_id: str) -> ServerResponse | None:
        """Get server details. Returns None if not found."""
        try:
            result = await self._request("GET", self._instance_path(f"/servers/{server_id}"))
        except ScalewayError as e:
            if e.status == 404:
                return None
            raise
        return result.get("server") if result else None

    async def server_action(self, server_id: str, action: str) -> None:
        """Execute an action on a server (poweron, poweroff, terminate)."""
        await self._request(
            "POST",
            self._instance_path(f"/servers/{server_id}/action"),
            json={"action": action},
        )

    async def delete_server(self, server_id: str) -> None:
        """Delete a server."""
        await self._request("DELETE", self._instance_path(f"/servers/{server_id}"))

    # =========================================================================
    # Block Storage (SBS) volumes
    # =========================================================================

    async def delete_block_volume(self, volume_id: str) -> None:
        """Delete an SBS volume via the Block Storage API."""
        try:
            await self._request("DELETE", self._block_path(f"/volumes/{volume_id}"))
        except ScalewayError as e:
            if e.status != 404:
                raise

    # =========================================================================
    # Images
    # =========================================================================

    async def list_images(
        self, *, public: bool = True, arch: str = "x86_64",
    ) -> list[ImageResponse]:
        """List available OS images."""
        result = await self._request(
            "GET",
            self._instance_path("/images"),
            params={"public": str(public).lower(), "arch": arch, "per_page": 100},
        )
        return result.get("images", []) if result else []

    # =========================================================================
    # SSH Keys (IAM API — not zone-scoped)
    # =========================================================================

    async def list_ssh_keys(self) -> list[SSHKeyResponse]:
        """List all SSH keys."""
        result = await self._request("GET", "/iam/v1alpha1/ssh-keys")
        return result.get("ssh_keys", []) if result else []

    async def create_ssh_key(self, name: str, public_key: str, project_id: str | None = None) -> SSHKeyResponse:
        """Create an SSH key."""
        payload: dict[str, Any] = {"name": name, "public_key": public_key}
        if project_id:
            payload["project_id"] = project_id
        result = await self._request("POST", "/iam/v1alpha1/ssh-keys", json=payload)
        return result

    async def delete_ssh_key(self, key_id: str) -> None:
        """Delete an SSH key."""
        try:
            await self._request("DELETE", f"/iam/v1alpha1/ssh-keys/{key_id}")
        except ScalewayError as e:
            if e.status != 404:
                raise


# =============================================================================
# Utility
# =============================================================================


def get_secret_key(config: Scaleway | None = None) -> str:
    """Get Scaleway secret key from config or environment."""
    if config and config.secret_key:
        return config.secret_key
    if env_key := os.environ.get("SCW_SECRET_KEY"):
        return env_key
    raise ValueError(
        "Scaleway secret key not found. "
        "Pass secret_key= to Scaleway() or set SCW_SECRET_KEY env var."
    )


def get_project_id(config: Scaleway | None = None) -> str:
    """Get Scaleway project ID from config or environment."""
    if config and config.project_id:
        return config.project_id
    if env_id := os.environ.get("SCW_DEFAULT_PROJECT_ID"):
        return env_id
    raise ValueError(
        "Scaleway project ID not found. "
        "Pass project_id= to Scaleway() or set SCW_DEFAULT_PROJECT_ID env var."
    )
