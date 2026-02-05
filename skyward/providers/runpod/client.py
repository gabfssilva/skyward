"""Async HTTP client for RunPod API.

Uses httpx for async HTTP requests with Bearer token authentication.
Returns TypedDicts directly.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from skyward.retry import on_status_code, retry

from .types import (
    ClusterCreateParams,
    ClusterResponse,
    CpuPodCreateParams,
    GpuTypeResponse,
    PodCreateParams,
    PodResponse,
)

RUNPOD_API_BASE = "https://rest.runpod.io/v1"
RUNPOD_GRAPHQL_URL = "https://api.runpod.io/graphql"

# GraphQL query for GPU types (same as official SDK)
GPU_TYPES_QUERY = """
query GpuTypes {
  gpuTypes {
    id
    displayName
    memoryInGb
    secureCloud
    communityCloud
  }
}
"""

# GraphQL mutation for creating Instant Clusters
# Note: Pod type in cluster response has different fields than standalone Pod
CREATE_CLUSTER_MUTATION = """
mutation CreateCluster($input: CreateClusterInput!) {
  createCluster(input: $input) {
    id
    name
    pods {
      id
      desiredStatus
      clusterIp
      clusterIdx
    }
  }
}
"""

# GraphQL mutation for deleting Instant Clusters
DELETE_CLUSTER_MUTATION = """
mutation DeleteCluster($input: DeleteClusterInput!) {
  deleteCluster(input: $input)
}
"""

# GraphQL mutation for creating CPU-only pods
DEPLOY_CPU_POD_MUTATION = """
mutation DeployCpuPod($input: deployCpuPodInput!) {
  deployCpuPod(input: $input) {
    id
    desiredStatus
    imageName
    machineId
  }
}
"""


class RunPodError(Exception):
    """Error from RunPod API."""


class RunPodClient:
    """Async HTTP client for RunPod API.

    Returns TypedDicts directly from API responses.

    Example:
        async with RunPodClient(api_key="...") as client:
            pod = await client.create_pod({...})
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "RunPodClient":
        self._client = httpx.AsyncClient(
            base_url=RUNPOD_API_BASE,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
        return self

    async def __aexit__(self, *_args: object) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Get the HTTP client, raising if not in context."""
        if not self._client:
            raise RuntimeError("RunPodClient must be used as async context manager")
        return self._client

    async def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Execute HTTP request and return JSON response."""
        try:
            resp = await self.client.request(method, path, json=json, params=params)
            resp.raise_for_status()
            return resp.json() if resp.content else None
        except httpx.HTTPStatusError as e:
            raise RunPodError(
                f"API error {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            raise RunPodError(f"Request failed ({type(e).__name__}): {e}") from e

    # =========================================================================
    # Pod Management
    # =========================================================================

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    async def create_pod(self, params: PodCreateParams) -> PodResponse:
        """Create a new pod."""
        result: PodResponse | None = await self._request("POST", "/pods", json=dict(params))
        if not result:
            raise RunPodError("Failed to create pod: empty response")
        return result

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    async def get_pod(self, pod_id: str) -> PodResponse | None:
        """Get pod details. Returns None if not found."""
        try:
            result: PodResponse | None = await self._request("GET", f"/pods/{pod_id}")
            return result
        except RunPodError as e:
            if "404" in str(e):
                return None
            raise

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    async def list_pods(self) -> list[PodResponse]:
        """List all pods."""
        result: list[PodResponse] | None = await self._request("GET", "/pods")
        return result or []

    async def stop_pod(self, pod_id: str) -> None:
        """Stop a pod (pause)."""
        await self._request("POST", f"/pods/{pod_id}/stop")

    async def terminate_pod(self, pod_id: str) -> None:
        """Terminate a pod (destroy)."""
        await self._request("DELETE", f"/pods/{pod_id}")

    # =========================================================================
    # GraphQL API
    # =========================================================================

    async def _graphql(
        self,
        query: str,
        variables: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a GraphQL query/mutation and return the data."""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    RUNPOD_GRAPHQL_URL,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"query": query, "variables": variables or {}},
                )
                resp.raise_for_status()
                data = resp.json()

            if "errors" in data:
                raise RunPodError(f"GraphQL error: {data['errors']}")

            return data.get("data", {})
        except httpx.HTTPStatusError as e:
            raise RunPodError(
                f"API error {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            raise RunPodError(f"Request failed ({type(e).__name__}): {e}") from e

    # =========================================================================
    # GPU Types (via GraphQL - no REST endpoint available)
    # =========================================================================

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    async def get_gpu_types(self) -> list[GpuTypeResponse]:
        """Get available GPU types via GraphQL API."""
        data = await self._graphql(GPU_TYPES_QUERY)
        gpu_types: list[GpuTypeResponse] = data.get("gpuTypes", [])
        return gpu_types

    # =========================================================================
    # Instant Clusters (via GraphQL)
    # =========================================================================

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    async def create_cluster(self, params: ClusterCreateParams) -> ClusterResponse:
        """Create an Instant Cluster with multiple pods.

        Instant Clusters provide high-speed networking (1600-3200 Gbps)
        between pods for distributed ML training.

        Args:
            params: Cluster creation parameters including gpuTypeId,
                    podCount, gpuCountPerPod, and type (TRAINING, APPLICATION, SLURM).

        Returns:
            ClusterResponse with cluster id, name, and pods list.
        """
        data = await self._graphql(CREATE_CLUSTER_MUTATION, {"input": dict(params)})
        cluster: ClusterResponse = data["createCluster"]
        return cluster

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    async def delete_cluster(self, cluster_id: str) -> None:
        """Delete an Instant Cluster and all its pods.

        Args:
            cluster_id: The cluster ID to delete.
        """
        await self._graphql(DELETE_CLUSTER_MUTATION, {"input": {"clusterId": cluster_id}})

    # =========================================================================
    # CPU-only Pods (via GraphQL)
    # =========================================================================

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    async def create_cpu_pod(self, params: "CpuPodCreateParams") -> PodResponse:
        """Create a CPU-only pod via GraphQL.

        The REST API doesn't support gpuCount=0, so we use the
        deployCpuPod GraphQL mutation instead.

        Args:
            params: CPU pod creation parameters.

        Returns:
            PodResponse with pod id and status.
        """
        data = await self._graphql(DEPLOY_CPU_POD_MUTATION, {"input": dict(params)})
        pod: PodResponse = data["deployCpuPod"]
        return pod


# =============================================================================
# Utility Functions
# =============================================================================


def _read_config_file() -> str | None:
    """Read API key from ~/.runpod/config.toml if it exists."""
    config_path = os.path.expanduser("~/.runpod/config.toml")
    if not os.path.exists(config_path):
        return None
    try:
        import tomllib
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        return config.get("default", {}).get("api_key")
    except Exception:
        return None


def get_api_key(config_key: str | None = None) -> str:
    """Get RunPod API key from config, environment, or CLI config file.

    Precedence:
        1. config_key parameter
        2. RUNPOD_API_KEY environment variable
        3. ~/.runpod/config.toml (from `runpod config`)

    Args:
        config_key: API key from config (optional).

    Returns:
        API key string.

    Raises:
        ValueError: If no API key found.
    """
    api_key = config_key or os.environ.get("RUNPOD_API_KEY") or _read_config_file()
    if not api_key:
        raise ValueError(
            "RunPod API key not found. Set RUNPOD_API_KEY environment variable, "
            "pass api_key to RunPod config, or run `runpod config`."
        )
    return api_key


__all__ = [
    "RUNPOD_API_BASE",
    "RunPodClient",
    "RunPodError",
    "get_api_key",
]
