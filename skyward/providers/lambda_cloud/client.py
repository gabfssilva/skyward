"""Async HTTP client for Lambda Cloud API."""

from __future__ import annotations

import json as json_mod
import os
from typing import Any

import httpx

from skyward.infra.http import HttpError
from skyward.infra.retry import on_status_code, retry
from skyward.infra.throttle import Limiter, throttle
from skyward.observability.logger import logger

from .config import Lambda
from .types import (
    InstanceResponse,
    InstanceTypeEntry,
    LaunchResponse,
    SSHKeyResponse,
)

LAMBDA_API_BASE = "https://cloud.lambdalabs.com/api/v1"


class LambdaError(Exception):
    """Error from Lambda Cloud API."""

    def __init__(
        self,
        code: str,
        message: str,
        suggestion: str | None = None,
    ) -> None:
        self.code = code
        self.suggestion = suggestion
        super().__init__(message)


class LambdaClient:
    """Async HTTP client for Lambda Cloud API."""

    def __init__(self, api_key: str, config: Lambda | None = None) -> None:
        self._api_key = api_key
        self._config = config or Lambda()
        self._http = httpx.AsyncClient(
            base_url=LAMBDA_API_BASE,
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
            timeout=httpx.Timeout(self._config.request_timeout),
        )
        self._launch_limiter = Limiter(interval=12.0)
        self._log = logger.bind(provider="lambda", component="client")

    async def __aenter__(self) -> LambdaClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    async def close(self) -> None:
        await self._http.aclose()

    @retry(on=on_status_code(429, 503), max_attempts=10, base_delay=1.0)
    @throttle(interval=1.0)
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
        """Make API request, unwrap data envelope, raise LambdaError on failure."""
        self._log.debug("{method} {path}", method=method, path=path)
        try:
            result = await self._do_request(method, path, json, params)
        except HttpError as e:
            try:
                error_data = json_mod.loads(e.body)
                error = error_data.get("error", {})
                raise LambdaError(
                    code=error.get("code", "unknown"),
                    message=error.get("message", str(e)),
                    suggestion=error.get("suggestion"),
                ) from e
            except (json_mod.JSONDecodeError, KeyError, TypeError):
                raise LambdaError(code="unknown", message=str(e)) from e

        # Unwrap {"data": ...} envelope
        if isinstance(result, dict) and "data" in result:
            return result["data"]
        return result

    # =========================================================================
    # Instance Types
    # =========================================================================

    async def list_instance_types(self) -> dict[str, InstanceTypeEntry]:
        """List available instance types with specs and regional availability."""
        result = await self._request("GET", "/instance-types")
        return result if isinstance(result, dict) else {}

    # =========================================================================
    # Instance Management
    # =========================================================================

    async def launch_instances(
        self,
        *,
        region_name: str,
        instance_type_name: str,
        ssh_key_names: list[str],
        quantity: int = 1,
        name: str | None = None,
        user_data: str | None = None,
    ) -> list[str]:
        """Launch instances. Returns list of instance IDs."""
        await self._launch_limiter.acquire()
        try:
            body: dict[str, Any] = {
                "region_name": region_name,
                "instance_type_name": instance_type_name,
                "ssh_key_names": ssh_key_names,
                "file_system_names": [],
                "quantity": quantity,
            }
            if name:
                body["name"] = name
            if user_data:
                body["user_data"] = user_data

            self._log.debug(
                "Launching {n}x {type} in {region}",
                n=quantity, type=instance_type_name, region=region_name,
            )
            result: LaunchResponse = await self._request(
                "POST", "/instance-operations/launch", json=body,
            )
            return result["instance_ids"]
        finally:
            self._launch_limiter.release()

    async def list_instances(self) -> list[InstanceResponse]:
        """List all running instances."""
        result = await self._request("GET", "/instances")
        return result if isinstance(result, list) else []

    async def get_instance(self, instance_id: str) -> InstanceResponse | None:
        """Get single instance details. Returns None if not found."""
        try:
            return await self._request("GET", f"/instances/{instance_id}")
        except LambdaError as e:
            if "not found" in str(e).lower() or e.code == "instance_not_found":
                return None
            raise

    async def terminate_instances(self, instance_ids: list[str]) -> None:
        """Terminate instances (batch)."""
        if not instance_ids:
            return
        self._log.debug("Terminating {n} instances", n=len(instance_ids))
        await self._request(
            "POST",
            "/instance-operations/terminate",
            json={"instance_ids": instance_ids},
        )

    # =========================================================================
    # SSH Key Management
    # =========================================================================

    async def list_ssh_keys(self) -> list[SSHKeyResponse]:
        """List all SSH keys on this account."""
        result = await self._request("GET", "/ssh-keys")
        return result if isinstance(result, list) else []

    async def add_ssh_key(self, name: str, public_key: str) -> SSHKeyResponse:
        """Register a new SSH public key. Returns key info."""
        result: SSHKeyResponse = await self._request(
            "POST", "/ssh-keys",
            json={"name": name, "public_key": public_key},
        )
        return result

    async def delete_ssh_key(self, key_id: str) -> None:
        """Delete an SSH key."""
        await self._request("DELETE", f"/ssh-keys/{key_id}")


def get_api_key(config: Lambda) -> str:
    """Resolve Lambda API key from config or environment."""
    if config.api_key:
        return config.api_key
    if env_key := os.environ.get("LAMBDA_API_KEY"):
        return env_key
    raise ValueError(
        "Lambda Cloud API key not found. "
        "Set LAMBDA_API_KEY or pass api_key to Lambda()."
    )
