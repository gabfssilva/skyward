"""Async HTTP client for Lambda Cloud API."""

from __future__ import annotations

import os
from typing import Any

import httpx

from skyward.infra.retry import on_status_code, retry
from skyward.infra.throttle import throttle
from skyward.observability.logger import logger

from .types import (
    InstanceResponse,
    InstanceTypeEntry,
    LaunchResponse,
    SSHKeyResponse,
)

LAMBDA_API_BASE = "https://cloud.lambda.ai/api/v1"

log = logger.bind(provider="lambda", component="client")


class LambdaError(Exception):
    """Error from Lambda Cloud API."""


class LambdaClient:
    """Async HTTP client for Lambda Cloud API.

    Example
    -------
    ::

        async with LambdaClient(api_key="...") as client:
            types = await client.list_instance_types()
            launched = await client.launch_instances(...)
    """

    def __init__(self, api_key: str, *, timeout: int = 30) -> None:
        self._http = httpx.AsyncClient(
            base_url=LAMBDA_API_BASE,
            auth=httpx.BasicAuth(api_key, ""),
            headers={"Accept": "application/json"},
            timeout=httpx.Timeout(timeout),
        )

    async def __aenter__(self) -> LambdaClient:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self._http.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
    ) -> Any:
        log.debug("{method} {path}", method=method, path=path)
        resp = await self._http.request(method, path, json=json)
        if resp.status_code >= 400:
            log.warning(
                "API error {method} {path}: {status}",
                method=method, path=path, status=resp.status_code,
            )
            body = resp.text
            raise LambdaError(f"API error {resp.status_code}: {body}")
        return resp.json() if resp.content else None

    # =========================================================================
    # Instance Types
    # =========================================================================

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    @throttle(max_concurrent=5, interval=0.05)
    async def list_instance_types(self) -> dict[str, InstanceTypeEntry]:
        """List available instance types with pricing and capacity."""
        result = await self._request("GET", "/instance-types")
        return result.get("data", {}) if result else {}

    # =========================================================================
    # Instance Operations
    # =========================================================================

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    @throttle(max_concurrent=5, interval=0.05)
    async def launch_instances(
        self,
        *,
        region_name: str,
        instance_type_name: str,
        ssh_key_names: list[str],
        quantity: int = 1,
        name: str | None = None,
    ) -> LaunchResponse:
        """Launch one or more instances."""
        payload: dict[str, Any] = {
            "region_name": region_name,
            "instance_type_name": instance_type_name,
            "ssh_key_names": ssh_key_names,
            "quantity": quantity,
        }
        if name:
            payload["name"] = name
        result = await self._request("POST", "/instance-operations/launch", json=payload)
        if not result:
            raise LambdaError("Failed to launch instances: empty response")
        return result["data"]

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    @throttle(max_concurrent=5, interval=0.05)
    async def get_instance(self, instance_id: str) -> InstanceResponse | None:
        """Get instance details. Returns None if not found."""
        try:
            result = await self._request("GET", f"/instances/{instance_id}")
            return result["data"] if result else None
        except LambdaError as e:
            if "404" in str(e):
                return None
            raise

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    @throttle(max_concurrent=5, interval=0.05)
    async def terminate_instances(self, instance_ids: list[str]) -> None:
        """Terminate one or more instances."""
        log.debug("Terminating instances {ids}", ids=instance_ids)
        await self._request(
            "POST", "/instance-operations/terminate",
            json={"instance_ids": instance_ids},
        )

    # =========================================================================
    # SSH Key Management
    # =========================================================================

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    async def list_ssh_keys(self) -> list[SSHKeyResponse]:
        """List all SSH keys."""
        result = await self._request("GET", "/ssh-keys")
        return result.get("data", []) if result else []

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    async def create_ssh_key(self, name: str, public_key: str) -> SSHKeyResponse:
        """Create an SSH key."""
        result = await self._request(
            "POST", "/ssh-keys",
            json={"name": name, "public_key": public_key},
        )
        if not result:
            raise LambdaError("Failed to create SSH key: empty response")
        return result["data"]

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    async def delete_ssh_key(self, key_id: str) -> None:
        """Delete an SSH key."""
        log.debug("Deleting SSH key {id}", id=key_id)
        await self._request("DELETE", f"/ssh-keys/{key_id}")


# =============================================================================
# Utility Functions
# =============================================================================


def get_api_key(config_key: str | None = None) -> str:
    """Get Lambda API key from config or environment.

    Parameters
    ----------
    config_key
        Explicit API key from config. Takes precedence over env var.

    Returns
    -------
    str
        The resolved API key.

    Raises
    ------
    ValueError
        If no API key found.
    """
    if api_key := config_key or os.environ.get("LAMBDA_API_KEY"):
        return api_key
    raise ValueError(
        "Lambda API key not found. Set LAMBDA_API_KEY environment variable "
        "or pass api_key to LambdaCloud config."
    )
