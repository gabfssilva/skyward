"""TensorDock API v2 client.

Uses the standard HttpClient + BearerAuth pattern. The api_token doubles
as the Bearer token for the v2 API at dashboard.tensordock.com.
"""

from __future__ import annotations

from typing import Any

from skyward.infra.http import BearerAuth, HttpClient, HttpError
from skyward.infra.retry import on_status_code, retry
from skyward.infra.throttle import throttle
from skyward.observability.logger import logger

from .types import (
    Location,
    V2AuthTestResponse,
    V2InstanceResponse,
)

log = logger.bind(component="tensordock")

TENSORDOCK_API_BASE = "https://dashboard.tensordock.com"


class TensorDockError(Exception):
    """Error from TensorDock API."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def _wrap(e: HttpError) -> TensorDockError:
    return TensorDockError(str(e), status=e.status)


class TensorDockClient:
    """Async client for TensorDock v2 API."""

    def __init__(self, api_token: str, *, timeout: int = 30) -> None:
        self._http = HttpClient(
            TENSORDOCK_API_BASE,
            BearerAuth(api_token),
            timeout=timeout,
        )
        self._public = HttpClient(TENSORDOCK_API_BASE, timeout=timeout)

    async def close(self) -> None:
        await self._http.close()
        await self._public.close()

    async def __aenter__(self) -> TensorDockClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    # ── Discovery (public, no auth) ──────────────────────────────────

    @retry(on=on_status_code(429, 500, 503), max_attempts=3, base_delay=2.0)
    @throttle(max_concurrent=2, interval=1.0)
    async def list_locations(self) -> list[Location]:
        """List all locations with GPU availability. No auth required."""
        try:
            data = await self._public.request("GET", "/api/v2/locations")
        except HttpError as e:
            raise _wrap(e) from e
        return data.get("data", {}).get("locations", [])

    # ── Auth ─────────────────────────────────────────────────────────

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    async def test_auth(self) -> bool:
        """Verify credentials via POST /api/v2/auth/test."""
        try:
            data: V2AuthTestResponse = await self._http.request(
                "POST", "/api/v2/auth/test",
            )
        except HttpError:
            return False
        return data.get("success", False)

    # ── Instances ────────────────────────────────────────────────────

    @retry(on=on_status_code(429, 503), max_attempts=5, base_delay=1.0)
    @throttle(max_concurrent=2, interval=1.0)
    async def create_instance(
        self,
        *,
        name: str,
        image: str,
        gpu_model: str,
        gpu_count: int,
        vcpus: int,
        ram_gb: int,
        storage_gb: int,
        location_id: str,
        ssh_key: str,
        cloud_init: dict[str, Any] | None = None,
        dedicated_ip: bool = False,
    ) -> V2InstanceResponse:
        """Create a VM via POST /api/v2/instances with location-based placement."""
        attributes: dict[str, Any] = {
            "name": name,
            "type": "virtualmachine",
            "image": image,
            "resources": {
                "vcpu_count": vcpus,
                "ram_gb": ram_gb,
                "storage_gb": storage_gb,
                "gpus": {gpu_model: {"count": gpu_count}},
            },
            "location_id": location_id,
            "ssh_key": ssh_key,
            "useDedicatedIp": dedicated_ip,
        }
        if cloud_init:
            attributes["cloud_init"] = cloud_init

        body = {"data": {"type": "virtualmachine", "attributes": attributes}}

        try:
            data = await self._http.request("POST", "/api/v2/instances", json=body)
        except HttpError as e:
            raise _wrap(e) from e
        return data.get("data", data)

    @retry(on=on_status_code(429, 503), max_attempts=5, base_delay=1.0)
    @throttle(max_concurrent=2, interval=1.0)
    async def get_instance(self, instance_id: str) -> V2InstanceResponse | None:
        """Get instance details. Returns None if not found."""
        try:
            data = await self._http.request("GET", f"/api/v2/instances/{instance_id}")
        except HttpError as e:
            if e.status == 404:
                return None
            raise _wrap(e) from e
        return data.get("data", data) if data else None

    @retry(on=on_status_code(429, 503), max_attempts=5, base_delay=1.0)
    @throttle(max_concurrent=2, interval=1.0)
    async def delete_instance(self, instance_id: str) -> None:
        """Delete an instance."""
        try:
            await self._http.request("DELETE", f"/api/v2/instances/{instance_id}")
        except HttpError as e:
            if e.status == 404:
                log.warning("Instance {id} already deleted", id=instance_id)
                return
            raise _wrap(e) from e

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    async def start_instance(self, instance_id: str) -> None:
        """Start a stopped instance."""
        try:
            await self._http.request("POST", f"/api/v2/instances/{instance_id}/start")
        except HttpError as e:
            raise _wrap(e) from e

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    async def stop_instance(self, instance_id: str) -> None:
        """Stop a running instance."""
        try:
            await self._http.request("POST", f"/api/v2/instances/{instance_id}/stop")
        except HttpError as e:
            raise _wrap(e) from e
