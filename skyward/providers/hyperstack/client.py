"""Async HTTP client for Hyperstack InfraHub API."""

from __future__ import annotations

import os
from typing import Any

import httpx

from skyward.infra.http import HttpError
from skyward.infra.retry import on_status_code, retry
from skyward.infra.throttle import throttle
from skyward.observability.logger import logger

from .config import Hyperstack
from .types import (
    AccessKeyResponse,
    CreateVMPayload,
    CreateVMResponse,
    EnvironmentResponse,
    FlavorResponse,
    ImageResponse,
    KeypairResponse,
    PricebookEntry,
    VMResponse,
)

HYPERSTACK_API_BASE = "https://infrahub-api.nexgencloud.com/v1"


class HyperstackError(Exception):
    """Error from Hyperstack API."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


# =============================================================================
# Async Client
# =============================================================================


class HyperstackClient:
    """Async HTTP client for Hyperstack InfraHub API."""

    def __init__(self, api_key: str, config: Hyperstack | None = None) -> None:
        self._api_key = api_key
        self._config = config or Hyperstack()
        self._http = httpx.AsyncClient(
            base_url=HYPERSTACK_API_BASE,
            headers={
                "api_key": api_key,
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(self._config.request_timeout),
        )
        self._log = logger.bind(provider="hyperstack", component="client")

    async def __aenter__(self) -> HyperstackClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    async def close(self) -> None:
        await self._http.aclose()

    @retry(on=on_status_code(429, 503), max_attempts=10, base_delay=0.5)
    @throttle(max_concurrent=5, interval=0.12)
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
            return await self._do_request(method, path, json, params)
        except HttpError as e:
            self._log.warning(
                "API error {method} {path}: {status}",
                method=method,
                path=path,
                status=e.status,
            )
            raise HyperstackError(f"API error {e.status}: {e.body}", status=e.status) from e

    # =========================================================================
    # Environments
    # =========================================================================

    async def create_environment(self, name: str, region: str) -> EnvironmentResponse:
        """Create an environment in a region."""
        result = await self._request(
            "POST",
            "/core/environments",
            json={"name": name, "region": region},
        )
        self._log.debug("create_environment response={r}", r=result)
        return result.get("environment", result)

    async def list_environments(self) -> list[EnvironmentResponse]:
        """List all environments."""
        result = await self._request("GET", "/core/environments")
        return result.get("environments", []) if result else []

    async def delete_environment(self, env_id: int) -> None:
        """Delete an environment (cascades keypairs and VMs)."""
        await self._request("DELETE", f"/core/environments/{env_id}")

    # =========================================================================
    # Keypairs
    # =========================================================================

    async def import_keypair(
        self, env_name: str, name: str, public_key: str,
    ) -> KeypairResponse:
        """Import an SSH keypair into an environment."""
        result = await self._request(
            "POST",
            "/core/keypairs",
            json={
                "name": name,
                "environment_name": env_name,
                "public_key": public_key,
            },
        )
        return result.get("keypair", result)

    async def list_keypairs(self) -> list[KeypairResponse]:
        """List all keypairs."""
        result = await self._request("GET", "/core/keypairs")
        return result.get("keypairs", []) if result else []

    # =========================================================================
    # Flavors & Images
    # =========================================================================

    async def list_flavors(self, region: str | None = None) -> list[FlavorResponse]:
        """List available hardware flavors.

        The API returns ``{"data": [{"gpu": ..., "region_name": ..., "flavors": [...]}]}``.
        We flatten the nested groups into a single list.
        """
        params = {"region": region} if region else None
        result = await self._request("GET", "/core/flavors", params=params)
        if not result:
            return []
        self._log.debug(
            "Flavors response keys={keys} type={t}",
            keys=list(result.keys()) if isinstance(result, dict) else "N/A",
            t=type(result).__name__,
        )
        data = result.get("data", [])
        self._log.debug(
            "Flavors data entries={n} first={first}",
            n=len(data),
            first=data[0] if data else None,
        )
        flavors = [
            flavor
            for group in data
            for flavor in group.get("flavors", [])
            if flavor.get("stock_available", True)
        ]
        self._log.debug("Flattened flavors count={n}", n=len(flavors))
        return flavors

    async def list_images(self, region: str | None = None) -> list[ImageResponse]:
        """List available OS images.

        The API returns ``{"images": [{"type": ..., "region_name": ..., "images": [...]}]}``.
        We flatten the nested groups into a single list.
        """
        params = {"region": region} if region else None
        result = await self._request("GET", "/core/images", params=params)
        if not result:
            return []
        groups = result.get("images", [])
        return [
            image
            for group in groups
            for image in group.get("images", [])
        ]

    # =========================================================================
    # Virtual Machines
    # =========================================================================

    async def create_vms(self, payload: CreateVMPayload) -> CreateVMResponse:
        """Create one or more VMs."""
        result = await self._request("POST", "/core/virtual-machines", json=dict(payload))
        return result or {}

    async def get_vm(self, vm_id: int) -> VMResponse | None:
        """Get VM details. Returns None if not found."""
        try:
            result = await self._request("GET", f"/core/virtual-machines/{vm_id}")
        except HyperstackError as e:
            if e.status == 404:
                return None
            raise
        if not result:
            return None
        return result.get("virtual_machine", result.get("instance", result))

    async def list_vms(self) -> list[VMResponse]:
        """List all VMs."""
        result = await self._request("GET", "/core/virtual-machines")
        return result.get("virtual_machines", []) if result else []

    async def delete_vm(self, vm_id: int) -> None:
        """Delete a VM."""
        await self._request("DELETE", f"/core/virtual-machines/{vm_id}")

    async def add_firewall_rule(
        self,
        vm_id: int,
        port_min: int,
        port_max: int,
        protocol: str = "tcp",
        remote_ip: str = "0.0.0.0/0",
    ) -> None:
        """Add a firewall (security group) rule to a VM."""
        await self._request(
            "POST",
            f"/core/virtual-machines/{vm_id}/sg-rules",
            json={
                "remote_ip_prefix": remote_ip,
                "direction": "ingress",
                "ethertype": "IPv4",
                "protocol": protocol,
                "port_range_min": port_min,
                "port_range_max": port_max,
            },
        )

    # =========================================================================
    # Object Storage Access Keys
    # =========================================================================

    async def create_access_key(self, region: str = "CANADA-1") -> AccessKeyResponse:
        """Create an object storage access key.

        The API returns fields at the top level (no wrapper key).
        """
        result = await self._request(
            "POST", "/object-storage/access-keys",
            json={"region": region},
        )
        return result

    async def delete_access_key(self, access_key_id: int) -> None:
        """Delete an object storage access key."""
        await self._request("DELETE", f"/object-storage/access-keys/{access_key_id}")

    # =========================================================================
    # Pricing
    # =========================================================================

    async def get_pricebook(self) -> list[PricebookEntry]:
        """Get hourly GPU pricing.

        The API returns a raw JSON array (not wrapped in a dict).
        """
        result = await self._request("GET", "/pricebook")
        self._log.debug(
            "Pricebook response type={t} len={n} first={first}",
            t=type(result).__name__,
            n=len(result) if isinstance(result, list) else "N/A",
            first=result[0] if isinstance(result, list) and result else None,
        )
        return result if isinstance(result, list) else []


# =============================================================================
# Utility
# =============================================================================


def get_api_key(config: Hyperstack | None = None) -> str:
    """Get Hyperstack API key from config or environment."""
    if config and config.api_key:
        return config.api_key
    if env_key := os.environ.get("HYPERSTACK_API_KEY"):
        return env_key
    raise ValueError(
        "Hyperstack API key not found. "
        "Pass api_key= to Hyperstack() or set HYPERSTACK_API_KEY env var."
    )
