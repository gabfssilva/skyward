"""Async HTTP client for Novita.ai API."""

from __future__ import annotations

import os
from typing import Any

import httpx

from skyward.infra.http import HttpError
from skyward.infra.retry import on_status_code, retry
from skyward.infra.throttle import throttle
from skyward.observability.logger import logger

from .types import (
    ClusterResponse,
    CreateInstanceResponse,
    InstanceResponse,
    ProductResponse,
)

NOVITA_API_BASE = "https://api.novita.ai/gpu-instance/openapi/v1"


class NovitaError(Exception):
    """Error from Novita.ai API."""

    def __init__(self, message: str, *, status: int = 0) -> None:
        super().__init__(message)
        self.status = status


def get_api_key(config_key: str | None = None) -> str:
    """Get Novita.ai API key from config or environment.

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

    if env_key := os.environ.get("NOVITA_API_KEY"):
        return env_key

    raise ValueError(
        "Novita.ai API key not found. Set NOVITA_API_KEY or pass api_key to Novita()"
    )


class NovitaClient:
    """Async HTTP client for Novita.ai API."""

    def __init__(self, api_key: str, *, request_timeout: int = 30) -> None:
        self._api_key = api_key
        self._http = httpx.AsyncClient(
            base_url=NOVITA_API_BASE,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(request_timeout),
        )
        self._log = logger.bind(provider="novita", component="client")

    async def __aenter__(self) -> NovitaClient:
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
                method=method, path=path, status=e.status,
            )
            raise NovitaError(
                f"API error {e.status}: {e.body}", status=e.status,
            ) from e

    async def list_clusters(self) -> list[ClusterResponse]:
        """List available clusters/regions.

        Returns
        -------
        list[ClusterResponse]
            Available clusters.
        """
        result = await self._request("GET", "/clusters")
        return result.get("data", []) if result else []

    async def list_products(
        self,
        *,
        cluster_id: str | None = None,
        gpu_num: int | None = None,
        product_name: str | None = None,
        billing_method: str | None = None,
    ) -> list[ProductResponse]:
        """List GPU products from the catalog.

        Parameters
        ----------
        cluster_id
            Filter by cluster ID.
        gpu_num
            Filter by GPU count.
        product_name
            Filter by product name.
        billing_method
            Filter by billing method (e.g., ``"onDemand"``).

        Returns
        -------
        list[ProductResponse]
            Available GPU products.
        """
        params: dict[str, Any] = {}
        if cluster_id:
            params["clusterId"] = cluster_id
        if gpu_num is not None:
            params["gpuNum"] = gpu_num
        if product_name:
            params["productName"] = product_name
        if billing_method:
            params["billingMethod"] = billing_method

        result = await self._request("GET", "/products", params=params)
        return result.get("data", []) if result else []

    async def create_instance(
        self,
        *,
        product_id: str,
        name: str,
        image_url: str,
        gpu_num: int = 1,
        rootfs_size: int = 50,
        billing_mode: str = "onDemand",
        ports: list[dict[str, Any]] | None = None,
        envs: list[dict[str, str]] | None = None,
        command: str | None = None,
        cluster_id: str | None = None,
        min_cuda_version: str | None = None,
    ) -> str:
        """Create a new GPU instance.

        Parameters
        ----------
        product_id
            GPU product to provision.
        name
            Instance display name.
        image_url
            Docker image URL.
        gpu_num
            Number of GPUs.
        rootfs_size
            Root filesystem size in GB.
        billing_mode
            Billing mode (``"onDemand"`` or ``"spot"``).
        ports
            Port mappings.
        envs
            Environment variables as ``[{"key": "K", "value": "V"}]``.
        command
            Startup command to run inside the container.
        cluster_id
            Target cluster ID.
        min_cuda_version
            Minimum CUDA version requirement.

        Returns
        -------
        str
            Created instance ID.

        Raises
        ------
        NovitaError
            If creation fails.
        """
        body: dict[str, Any] = {
            "productId": product_id,
            "name": name,
            "imageUrl": image_url,
            "gpuNum": gpu_num,
            "rootfsSize": rootfs_size,
            "billingMode": billing_mode,
        }
        if ports:
            body["ports"] = ports
        if envs:
            body["envs"] = envs
        if command:
            body["command"] = command
        if cluster_id:
            body["clusterId"] = cluster_id
        if min_cuda_version:
            body["minCudaVersion"] = min_cuda_version

        self._log.debug(
            "Creating instance: product={pid}, image={img}",
            pid=product_id, img=image_url,
        )
        result: CreateInstanceResponse | None = await self._request(
            "POST", "/gpu/instance/create", json=body,
        )

        if not result:
            raise NovitaError("Empty response from instance creation")

        if error := result.get("error") or result.get("message"):
            raise NovitaError(f"Instance creation failed: {error}")

        instance_id = result.get("instanceId") or result.get("id")
        if not instance_id:
            raise NovitaError(f"No instance ID in response: {result}")

        self._log.debug("Created instance {iid}", iid=instance_id)
        return instance_id

    async def get_instance(self, instance_id: str) -> InstanceResponse | None:
        """Get instance details.

        Parameters
        ----------
        instance_id
            Instance ID to query.

        Returns
        -------
        InstanceResponse | None
            Instance details, or ``None`` if not found.
        """
        try:
            result = await self._request(
                "GET", "/gpu/instance",
                params={"instanceId": instance_id},
            )
        except NovitaError as e:
            if e.status == 404:
                return None
            raise

        if not result:
            return None

        match result:
            case {"instance": dict() as inst}:
                return inst  # type: ignore[return-value]
            case dict() if "id" in result:
                return result  # type: ignore[return-value]
            case _:
                return None

    async def list_instances(
        self,
        *,
        page_size: int = 100,
        page_number: int = 1,
        status: str | None = None,
    ) -> list[InstanceResponse]:
        """List all instances.

        Parameters
        ----------
        page_size
            Results per page.
        page_number
            Page number (1-indexed).
        status
            Filter by status.

        Returns
        -------
        list[InstanceResponse]
            Instances matching the query.
        """
        params: dict[str, Any] = {
            "pageSize": page_size,
            "pageNumber": page_number,
        }
        if status:
            params["status"] = status

        result = await self._request("GET", "/gpu/instances", params=params)
        return result.get("data", result.get("instances", [])) if result else []

    async def delete_instance(self, instance_id: str) -> None:
        """Delete/terminate an instance.

        Parameters
        ----------
        instance_id
            Instance to terminate.
        """
        self._log.debug("Deleting instance {iid}", iid=instance_id)
        await self._request(
            "POST", "/gpu/instance/delete",
            json={"instanceId": instance_id},
        )

    async def stop_instance(self, instance_id: str) -> None:
        """Stop a running instance.

        Parameters
        ----------
        instance_id
            Instance to stop.
        """
        self._log.debug("Stopping instance {iid}", iid=instance_id)
        await self._request(
            "POST", "/gpu/instance/stop",
            json={"instanceId": instance_id},
        )
