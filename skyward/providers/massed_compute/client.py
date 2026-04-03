"""Async HTTP client for Massed Compute API."""

from __future__ import annotations

import os
from typing import Any

import httpx

from skyward.infra.http import HttpError
from skyward.infra.retry import on_status_code, retry
from skyward.infra.throttle import throttle
from skyward.observability.logger import logger

from .types import (
    ImageResponse,
    InstanceResponse,
    InventoryItem,
    SSHKeyResponse,
)

MASSED_API_BASE = "https://vm.massedcompute.com/api/v1"


class MassedComputeError(Exception):
    """Error from Massed Compute API."""

    def __init__(self, message: str, *, status: int = 0) -> None:
        super().__init__(message)
        self.status = status


def get_api_key(config_key: str | None = None) -> str:
    """Get Massed Compute API key from config or environment.

    Parameters
    ----------
    config_key
        Explicit API key from config. Checked first.

    Returns
    -------
    str
        Resolved API key.

    Raises
    ------
    ValueError
        If no API key is found.
    """
    if config_key:
        return config_key

    if env_key := os.environ.get("MASSED_API_KEY"):
        return env_key

    raise ValueError(
        "Massed Compute API key not found. Set MASSED_API_KEY or pass api_key to MassedCompute()"
    )


class MassedComputeClient:
    """Async HTTP client for Massed Compute API."""

    def __init__(self, api_key: str, *, request_timeout: int = 30) -> None:
        self._http = httpx.AsyncClient(
            base_url=MASSED_API_BASE,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            },
            timeout=httpx.Timeout(request_timeout),
        )
        self._log = logger.bind(provider="massed_compute", component="client")

    async def __aenter__(self) -> MassedComputeClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    async def close(self) -> None:
        await self._http.aclose()

    @retry(on=on_status_code(429, 503), max_attempts=10, base_delay=0.5)
    @throttle(max_concurrent=2, interval=1.0)
    async def _do_request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
    ) -> Any:
        resp = await self._http.request(method, path, json=json)
        if resp.status_code >= 400:
            raise HttpError(status=resp.status_code, body=resp.text)
        return resp.json() if resp.content else None

    async def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
    ) -> Any:
        self._log.debug("{method} {path}", method=method, path=path)
        try:
            return await self._do_request(method, path, json)
        except HttpError as e:
            self._log.warning(
                "API error {method} {path}: {status}",
                method=method, path=path, status=e.status,
            )
            raise MassedComputeError(
                f"API error {e.status}: {e.body}", status=e.status,
            ) from e

    async def gpu_inventory(self) -> dict[str, InventoryItem]:
        """List GPU inventory with pricing and availability.

        Returns
        -------
        dict[str, InventoryItem]
            Keyed by product name (e.g., ``"gpu_1x_a6000"``).
        """
        result = await self._request("GET", "/gpu-inventory")
        return result.get("gpu_inventory", {}) if result else {}

    async def list_images(self) -> list[ImageResponse]:
        """List available OS images."""
        result = await self._request("GET", "/images")
        return result.get("images", []) if result else []

    async def list_ssh_keys(self) -> list[SSHKeyResponse]:
        """List registered SSH keys."""
        result = await self._request("GET", "/ssh-keys")
        return result.get("sshKeys", []) if result else []

    async def create_ssh_key(self, name: str, public_key: str) -> SSHKeyResponse:
        """Register an SSH public key.

        Parameters
        ----------
        name
            Key name (alphanumeric only, no hyphens or special chars).
        public_key
            SSH public key content.

        Returns
        -------
        SSHKeyResponse
            Created key with ``id`` and ``name``.
        """
        result = await self._request(
            "POST", "/ssh-keys",
            json={"name": name, "publicKey": public_key},
        )
        return result["sshKey"]

    async def delete_ssh_key(self, key_id: str) -> None:
        """Delete a registered SSH key."""
        await self._request("DELETE", f"/ssh-keys/{key_id}")

    async def launch_instance(
        self,
        *,
        product_name: str,
        image_id: int,
        instance_name: str | None = None,
        ssh_key_names: tuple[str, ...] = (),
    ) -> str:
        """Launch a new VM instance.

        Parameters
        ----------
        product_name
            GPU product name (e.g., ``"gpu_1x_a6000"``).
        image_id
            OS image ID (``84`` = Ubuntu 22.04 w/ drivers).
        instance_name
            Display name for the instance.
        ssh_key_names
            SSH key names to inject at launch.

        Returns
        -------
        str
            Instance UUID.

        Raises
        ------
        MassedComputeError
            If launch fails.
        """
        body: dict[str, Any] = {
            "productName": product_name,
            "regionName": "any",
            "imageId": image_id,
        }
        if instance_name:
            body["instanceName"] = instance_name
        if ssh_key_names:
            body["sshKeys"] = list(ssh_key_names)

        self._log.debug(
            "Launching instance: product={product}, image={image}",
            product=product_name, image=image_id,
        )
        result = await self._request("POST", "/instance/launch", json=body)

        if not result or "response" not in result:
            raise MassedComputeError(f"No UUID in launch response: {result}")

        instance_uuid = result["response"]
        self._log.debug("Launched instance {uuid}", uuid=instance_uuid)
        return instance_uuid

    async def get_instance(self, instance_uuid: str) -> InstanceResponse | None:
        """Get instance details by UUID.

        Returns
        -------
        InstanceResponse | None
            Instance details, or ``None`` if not found.
        """
        try:
            result = await self._request("GET", f"/instance/{instance_uuid}")
        except MassedComputeError as e:
            if e.status == 404:
                return None
            raise

        if not result:
            return None

        instances = result.get("runningInstances", [])
        return instances[0] if instances else None

    async def terminate_instances(self, instance_uuids: tuple[str, ...]) -> None:
        """Terminate one or more instances.

        Parameters
        ----------
        instance_uuids
            UUIDs of instances to terminate.
        """
        if not instance_uuids:
            return

        self._log.debug(
            "Terminating {n} instances", n=len(instance_uuids),
        )
        await self._request(
            "POST", "/instance/terminate",
            json={"instanceUuids": list(instance_uuids)},
        )
