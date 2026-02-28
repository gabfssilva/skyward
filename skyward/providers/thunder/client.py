"""Async HTTP client for Thunder Compute API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from skyward.infra.http import HttpClient, HttpError
from skyward.infra.retry import on_status_code, retry
from skyward.observability.logger import logger

from .types import InstanceCreateResponse, InstanceListItem, PricingEntry, SSHKeyResponse

THUNDER_API_BASE = "https://api.thundercompute.com:8443/v1"


class ThunderError(Exception):
    """Error from Thunder Compute API."""


class ThunderClient:
    def __init__(self, http_client: HttpClient) -> None:
        self._http = http_client
        self._log = logger.bind(provider="thunder", component="client")

    async def close(self) -> None:
        await self._http.close()

    async def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        self._log.debug("{method} {path}", method=method, path=path)
        try:
            return await self._http.request(method, path, json=json, params=params)
        except HttpError as e:
            self._log.warning(
                "API error {method} {path}: {status}",
                method=method, path=path, status=e.status,
            )
            raise ThunderError(f"API error {e.status}: {e.body}") from e

    # =========================================================================
    # Pricing
    # =========================================================================

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    async def get_pricing(self) -> dict[str, PricingEntry]:
        """Get GPU pricing information."""
        result: dict[str, PricingEntry] | None = await self._request("GET", "/pricing")
        return result or {}

    # =========================================================================
    # SSH Key Management
    # =========================================================================

    async def list_ssh_keys(self) -> list[SSHKeyResponse]:
        """List all SSH keys registered on this account."""
        result: list[SSHKeyResponse] | None = await self._request("GET", "/keys/list")
        return result or []

    async def add_ssh_key(self, name: str, public_key: str) -> SSHKeyResponse:
        """Register a new SSH public key."""
        self._log.debug("Adding SSH key '{name}'", name=name)
        result: SSHKeyResponse | None = await self._request(
            "POST", "/keys/add", json={"name": name, "public_key": public_key},
        )
        if not result:
            raise ThunderError("Failed to add SSH key: empty response")
        return result

    async def delete_ssh_key(self, key_id: str) -> None:
        """Delete an SSH key."""
        await self._request("DELETE", f"/keys/{key_id}")

    # =========================================================================
    # Instance Management
    # =========================================================================

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    async def create_instance(
        self,
        cpu_cores: int,
        disk_size_gb: int,
        gpu_type: str,
        num_gpus: int,
        mode: str,
        template: str,
        public_key: str,
    ) -> InstanceCreateResponse:
        """Create a new GPU instance."""
        result: InstanceCreateResponse | None = await self._request(
            "POST",
            "/instances/create",
            json={
                "cpu_cores": cpu_cores,
                "disk_size_gb": disk_size_gb,
                "gpu_type": gpu_type,
                "num_gpus": num_gpus,
                "mode": mode,
                "template": template,
                "public_key": public_key,
            },
        )
        if not result:
            raise ThunderError("Failed to create instance: empty response")
        return result

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    async def list_instances(self) -> dict[str, InstanceListItem]:
        """List all instances on this account."""
        result: dict[str, InstanceListItem] | None = await self._request(
            "GET", "/instances/list",
        )
        return result or {}

    async def delete_instance(self, uuid: str) -> None:
        """Delete an instance."""
        self._log.debug("Deleting instance {uuid}", uuid=uuid)
        await self._request("POST", f"/instances/{uuid}/delete")

    async def add_key_to_instance(self, uuid: str, public_key: str) -> None:
        """Add an SSH public key to a running instance."""
        await self._request(
            "POST", f"/instances/{uuid}/add_key", json={"public_key": public_key},
        )


# =============================================================================
# Utility Functions
# =============================================================================


def get_api_key(config_key: str | None = None) -> str:
    """Resolve Thunder Compute API key.

    Resolution order: explicit config value, TNR_API_TOKEN env var,
    ~/.thunder/token file.

    Parameters
    ----------
    config_key
        Explicit API key from provider config. Used first if provided.

    Returns
    -------
    str
        The resolved API key.

    Raises
    ------
    ValueError
        If no API key can be found.
    """
    if config_key:
        return config_key

    if env_key := os.environ.get("TNR_API_TOKEN"):
        return env_key

    token_path = Path.home() / ".thunder" / "token"
    if token_path.exists():
        token = token_path.read_text().strip()
        if token:
            return token

    raise ValueError(
        "Thunder Compute API key not found. "
        "Set TNR_API_TOKEN environment variable, "
        "pass api_token to ThunderCompute(), "
        "or write token to ~/.thunder/token."
    )
