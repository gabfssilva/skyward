"""Async HTTP client for Vultr API."""

from __future__ import annotations

import os
from typing import Any

import httpx

from skyward.infra.retry import on_status_code, retry
from skyward.infra.throttle import throttle
from skyward.observability.logger import logger

from .types import (
    BareMetalCreateParams,
    BareMetalResponse,
    InstanceCreateParams,
    InstanceResponse,
    MetalPlanResponse,
    PlanResponse,
    SSHKeyResponse,
)

VULTR_API_BASE = "https://api.vultr.com/v2"

log = logger.bind(provider="vultr", component="client")


class VultrError(Exception):
    """Error from Vultr API."""


class VultrClient:
    """Async HTTP client for Vultr Cloud GPU and Bare Metal APIs.

    Example
    -------
    ::

        async with VultrClient(api_key="...") as client:
            plans = await client.list_gpu_plans()
            instance = await client.create_instance({...})
    """

    def __init__(self, api_key: str, *, timeout: int = 30) -> None:
        self._http = httpx.AsyncClient(
            base_url=VULTR_API_BASE,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout),
        )

    async def __aenter__(self) -> VultrClient:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self._http.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        log.debug("{method} {path}", method=method, path=path)
        resp = await self._http.request(method, path, json=json, params=params)
        if resp.status_code >= 400:
            log.warning(
                "API error {method} {path}: {status}",
                method=method, path=path, status=resp.status_code,
            )
            raise VultrError(f"API error {resp.status_code}: {resp.text}")
        return resp.json() if resp.content else None

    # =========================================================================
    # Cloud GPU Instances (/instances with vcg-* plans)
    # =========================================================================

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    @throttle(max_concurrent=5, interval=0.05)
    async def list_gpu_plans(self) -> list[PlanResponse]:
        """List available Cloud GPU plans (vcg-* type)."""
        result = await self._request("GET", "/plans", params={"type": "vcg"})
        return result.get("plans", []) if result else []

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    @throttle(max_concurrent=5, interval=0.05)
    async def create_instance(self, params: InstanceCreateParams) -> InstanceResponse:
        """Create a cloud GPU instance."""
        result = await self._request("POST", "/instances", json=dict(params))
        if not result:
            raise VultrError("Failed to create instance: empty response")
        return result["instance"]

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    @throttle(max_concurrent=5, interval=0.05)
    async def get_instance(self, instance_id: str) -> InstanceResponse | None:
        """Get cloud instance details. Returns None if not found."""
        try:
            result = await self._request("GET", f"/instances/{instance_id}")
            return result["instance"] if result else None
        except VultrError as e:
            if "404" in str(e):
                return None
            raise

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    @throttle(max_concurrent=5, interval=0.05)
    async def delete_instance(self, instance_id: str) -> None:
        """Delete a cloud instance."""
        log.debug("Deleting instance {id}", id=instance_id)
        await self._request("DELETE", f"/instances/{instance_id}")

    # =========================================================================
    # Bare Metal (/bare-metals with vbm-* plans)
    # =========================================================================

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    @throttle(max_concurrent=5, interval=0.05)
    async def list_metal_plans(self) -> list[MetalPlanResponse]:
        """List available bare metal plans."""
        result = await self._request("GET", "/plans-metal")
        return result.get("plans_metal", []) if result else []

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    @throttle(max_concurrent=5, interval=0.05)
    async def create_bare_metal(
        self, params: BareMetalCreateParams,
    ) -> BareMetalResponse:
        """Create a bare metal instance."""
        result = await self._request("POST", "/bare-metals", json=dict(params))
        if not result:
            raise VultrError("Failed to create bare metal: empty response")
        return result["bare_metal"]

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    @throttle(max_concurrent=5, interval=0.05)
    async def get_bare_metal(self, instance_id: str) -> BareMetalResponse | None:
        """Get bare metal instance details. Returns None if not found."""
        try:
            result = await self._request("GET", f"/bare-metals/{instance_id}")
            return result["bare_metal"] if result else None
        except VultrError as e:
            if "404" in str(e):
                return None
            raise

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    @throttle(max_concurrent=5, interval=0.05)
    async def delete_bare_metal(self, instance_id: str) -> None:
        """Delete a bare metal instance."""
        log.debug("Deleting bare metal {id}", id=instance_id)
        await self._request("DELETE", f"/bare-metals/{instance_id}")

    # =========================================================================
    # SSH Key Management
    # =========================================================================

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    async def list_ssh_keys(self) -> list[SSHKeyResponse]:
        """List all SSH keys."""
        result = await self._request("GET", "/ssh-keys")
        return result.get("ssh_keys", []) if result else []

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    async def create_ssh_key(self, name: str, ssh_key: str) -> SSHKeyResponse:
        """Create an SSH key."""
        result = await self._request(
            "POST", "/ssh-keys",
            json={"name": name, "ssh_key": ssh_key},
        )
        if not result:
            raise VultrError("Failed to create SSH key: empty response")
        return result["ssh_key"]


# =============================================================================
# Utility Functions
# =============================================================================


def get_api_key(config_key: str | None = None) -> str:
    """Get Vultr API key from config or environment.

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
    if api_key := config_key or os.environ.get("VULTR_API_KEY"):
        return api_key
    raise ValueError(
        "Vultr API key not found. Set VULTR_API_KEY environment variable "
        "or pass api_key to Vultr config."
    )
