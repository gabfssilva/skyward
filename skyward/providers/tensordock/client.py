"""TensorDock API client.

Uses aiohttp directly because TensorDock v0 API authenticates via
query params (api_key + api_token) and deploy uses form-encoded POST,
neither of which fit the standard HttpClient + BearerAuth pattern.
"""

from __future__ import annotations

from typing import Any

import aiohttp

from skyward.infra.retry import on_status_code, retry
from skyward.infra.throttle import throttle
from skyward.observability.logger import logger

from .types import (
    AuthTestResponse,
    DeployResponse,
    HostnodeResponse,
    VmDetails,
    VmGetResponse,
    VmListResponse,
)

log = logger.bind(component="tensordock")

TENSORDOCK_API_BASE = "https://marketplace.tensordock.com"


class TensorDockError(Exception):
    """Error from TensorDock API."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class TensorDockClient:
    """Async client for TensorDock Marketplace v0 API."""

    def __init__(
        self,
        api_key: str,
        api_token: str,
        *,
        timeout: int = 30,
    ) -> None:
        self._api_key = api_key
        self._api_token = api_token
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: aiohttp.ClientSession | None = None

    def _auth_params(self) -> dict[str, str]:
        return {"api_key": self._api_key, "api_token": self._api_token}

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def __aenter__(self) -> TensorDockClient:
        await self._ensure_session()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # -----------------------------------------------------------------
    # Internal request helpers
    # -----------------------------------------------------------------

    @retry(on=on_status_code(429, 503), max_attempts=5, base_delay=1.0)
    @throttle(max_concurrent=2, interval=1.0)
    async def _get(self, path: str, **extra_params: Any) -> Any:
        session = await self._ensure_session()
        params = {**self._auth_params(), **extra_params}
        url = f"{TENSORDOCK_API_BASE}{path}"
        async with session.get(url, params=params) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise TensorDockError(
                    f"GET {path} → {resp.status}: {body}",
                    status=resp.status,
                )
            return await resp.json()

    @retry(on=on_status_code(429, 503), max_attempts=5, base_delay=1.0)
    @throttle(max_concurrent=2, interval=1.0)
    async def _post_json(self, path: str, body: dict[str, Any]) -> Any:
        session = await self._ensure_session()
        payload = {**self._auth_params(), **body}
        url = f"{TENSORDOCK_API_BASE}{path}"
        async with session.post(url, json=payload) as resp:
            if resp.status >= 400:
                text = await resp.text()
                raise TensorDockError(
                    f"POST {path} → {resp.status}: {text}",
                    status=resp.status,
                )
            return await resp.json()

    @retry(on=on_status_code(429, 503), max_attempts=5, base_delay=1.0)
    @throttle(max_concurrent=2, interval=1.0)
    async def _post_form(self, path: str, body: dict[str, Any]) -> Any:
        session = await self._ensure_session()
        payload = {**self._auth_params(), **body}
        url = f"{TENSORDOCK_API_BASE}{path}"
        async with session.post(url, data=payload) as resp:
            if resp.status >= 400:
                text = await resp.text()
                raise TensorDockError(
                    f"POST {path} → {resp.status}: {text}",
                    status=resp.status,
                )
            return await resp.json()

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    async def test_auth(self) -> bool:
        """Verify credentials via /api/v0/auth/test."""
        result: AuthTestResponse = await self._post_json("/api/v0/auth/test", {})
        return result.get("success", False)

    async def list_hostnodes(self) -> dict[str, HostnodeResponse]:
        """List hostnodes with GPU stock, pricing, and locations.

        No authentication required.
        """
        session = await self._ensure_session()
        url = f"{TENSORDOCK_API_BASE}/api/v0/client/deploy/hostnodes"
        async with session.get(url) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise TensorDockError(
                    f"list_hostnodes → {resp.status}: {body}",
                    status=resp.status,
                )
            data = await resp.json()
        data.pop("success", None)
        return data

    async def deploy_vm(
        self,
        *,
        hostnode: str,
        gpu_model: str,
        gpu_count: int,
        vcpus: int,
        ram: int,
        storage: int,
        password: str,
        name: str,
        operating_system: str,
        internal_ports: str,
        external_ports: str,
        cloudinit_script: str,
    ) -> DeployResponse:
        """Deploy a VM on a specific hostnode.

        Uses form-encoded POST body as required by TensorDock v0 API.
        """
        body = {
            "password": password,
            "name": name,
            "gpu_count": str(gpu_count),
            "gpu_model": gpu_model,
            "vcpus": str(vcpus),
            "ram": str(ram),
            "external_ports": external_ports,
            "internal_ports": internal_ports,
            "hostnode": hostnode,
            "storage": str(storage),
            "operating_system": operating_system,
            "cloudinit_script": cloudinit_script,
        }
        result: DeployResponse = await self._post_form(
            "/api/v0/client/deploy/single", body,
        )
        if not result.get("success"):
            raise TensorDockError(
                f"Deploy failed: {result.get('error', 'unknown error')}",
            )
        return result

    async def get_vm(self, server_id: str) -> VmDetails | None:
        """Get single VM details. Returns None if not found."""
        try:
            result: VmGetResponse = await self._post_json(
                "/api/v0/client/get/single", {"server": server_id},
            )
        except TensorDockError:
            return None
        if not result.get("success"):
            return None
        return result.get("virtualmachine")

    async def list_vms(self) -> dict[str, VmDetails]:
        """List all deployed VMs."""
        result: VmListResponse = await self._post_json(
            "/api/v0/client/list", {},
        )
        if not result.get("success"):
            return {}
        return result.get("virtualmachines", {})

    async def delete_vm(self, server_id: str) -> None:
        """Delete a VM."""
        result = await self._post_json(
            "/api/v0/client/delete/single", {"server": server_id},
        )
        if not result.get("success"):
            log.warning(
                "Delete VM {id} may have failed: {r}", id=server_id, r=result,
            )

    async def start_vm(self, server_id: str) -> None:
        """Start a stopped VM."""
        await self._get("/api/v0/client/start/single", server=server_id)

    async def stop_vm(self, server_id: str) -> None:
        """Stop a running VM."""
        await self._get("/api/v0/client/stop/single", server=server_id)
