"""Async HTTP client for RunPod API."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import RunPod

from skyward.infra.http import BearerAuth, HttpClient, HttpError
from skyward.infra.retry import on_status_code, retry
from skyward.observability.logger import logger

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
query GpuTypes($minCudaVersion: String, $secureCloud: Boolean) {
  gpuTypes {
    id
    displayName
    memoryInGb
    secureCloud
    communityCloud
    lowestPrice(input: {
      gpuCount: 1,
      minCudaVersion: $minCudaVersion,
      secureCloud: $secureCloud
    }) {
      minimumBidPrice
      uninterruptablePrice
      stockStatus
      totalCount
      rentedCount
    }
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


DEPLOY_ON_DEMAND_MUTATION = """
mutation DeployOnDemand($input: PodFindAndDeployOnDemandInput) {
  podFindAndDeployOnDemand(input: $input) {
    id
    desiredStatus
    imageName
    costPerHr
    gpuCount
    vcpuCount
    memoryInGb
    machine { dataCenterId gpuDisplayName }
  }
}
"""

DEPLOY_SPOT_MUTATION = """
mutation DeploySpot($input: PodRentInterruptableInput!) {
  podRentInterruptable(input: $input) {
    id
    desiredStatus
    imageName
    costPerHr
    gpuCount
    vcpuCount
    memoryInGb
    machine { dataCenterId gpuDisplayName }
  }
}
"""

MYSELF_QUERY = """
query { myself { id pubKey } }
"""

REGISTRY_AUTHS_QUERY = """
query { myself { containerRegistryCreds { id name } } }
"""

UPDATE_USER_SETTINGS_MUTATION = """
mutation Mutation($input: UpdateUserSettingsInput) {
  updateUserSettings(input: $input) {
    id
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

    def __init__(self, api_key: str, config: RunPod | None = None) -> None:
        self._api_key = api_key
        self._config = config or RunPod()
        self._log = logger.bind(provider="runpod", component="client")
        self._http = HttpClient(
            RUNPOD_API_BASE,
            BearerAuth(api_key),
            timeout=self._config.request_timeout,
            default_headers={"Content-Type": "application/json"},
        )

    async def __aenter__(self) -> RunPodClient:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self._http.close()

    async def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        try:
            return await self._http.request(method, path, json=json, params=params)
        except HttpError as e:
            self._log.warning(
                "API error {method} {path}: {status}",
                method=method, path=path, status=e.status,
            )
            raise RunPodError(f"API error {e.status}: {e.body}") from e

    # =========================================================================
    # Pod Management
    # =========================================================================

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    async def create_pod(self, params: PodCreateParams) -> PodResponse:
        """Create a new pod via REST API."""
        self._log.debug("Creating pod")
        result: PodResponse | None = await self._request("POST", "/pods", json=dict(params))
        if not result:
            raise RunPodError("Failed to create pod: empty response")
        return result

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    async def deploy_gpu_pod(
        self,
        *,
        name: str,
        image_name: str,
        gpu_type_id: str,
        gpu_count: int = 1,
        cloud_type: str = "SECURE",
        container_disk_gb: int = 50,
        volume_gb: int = 20,
        volume_mount_path: str = "/workspace",
        ports: str = "22/tcp",
        interruptible: bool = False,
        data_center_id: str | None = None,
        deploy_cost: float | None = None,
        spot_price: float | None = None,
        allowed_cuda_versions: list[str] | None = None,
        container_registry_auth_id: str | None = None,
    ) -> PodResponse:
        """Deploy a GPU pod via GraphQL."""
        input_vars: dict[str, Any] = {
            "name": name,
            "imageName": image_name,
            "gpuTypeId": gpu_type_id,
            "gpuCount": gpu_count,
            "cloudType": cloud_type,
            "containerDiskInGb": container_disk_gb,
            "volumeInGb": volume_gb,
            "volumeMountPath": volume_mount_path,
            "ports": ports,
            "startSsh": True,
        }

        if data_center_id:
            input_vars["dataCenterId"] = data_center_id
        if allowed_cuda_versions:
            input_vars["allowedCudaVersions"] = allowed_cuda_versions
        if container_registry_auth_id:
            input_vars["containerRegistryAuthId"] = container_registry_auth_id

        if interruptible:
            input_vars["bidPerGpu"] = deploy_cost or spot_price or 0.0
            mutation = DEPLOY_SPOT_MUTATION
            result_key = "podRentInterruptable"
        else:
            if deploy_cost:
                input_vars["deployCost"] = deploy_cost
            mutation = DEPLOY_ON_DEMAND_MUTATION
            result_key = "podFindAndDeployOnDemand"

        self._log.debug("Deploying GPU pod via GraphQL ({key})", key=result_key)
        data = await self._graphql(mutation, {"input": input_vars})

        pod: PodResponse | None = data.get(result_key)
        if not pod:
            raise RunPodError(f"Failed to deploy pod: empty response from {result_key}")
        return pod

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
        self._log.debug("Terminating pod {pod_id}", pod_id=pod_id)
        await self._request("DELETE", f"/pods/{pod_id}")

    # =========================================================================
    # GraphQL API
    # =========================================================================

    async def _graphql(
        self,
        query: str,
        variables: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        op_name = query.strip().split("(")[0].split("{")[0].strip().split()[-1]
        self._log.debug("Executing GraphQL: {op}", op=op_name)
        try:
            async with HttpClient(
                RUNPOD_GRAPHQL_URL,
                BearerAuth(self._api_key),
                timeout=self._config.request_timeout * 2,
                default_headers={"Content-Type": "application/json"},
            ) as http:
                data = await http.request(
                    "POST", "", json={"query": query, "variables": variables or {}}
                )

            match data:
                case dict() as d if "errors" in d:
                    self._log.warning(
                        "GraphQL error in {op}: {errors}",
                        op=op_name, errors=d["errors"],
                    )
                    raise RunPodError(f"GraphQL error: {d['errors']}")
                case dict() as d:
                    return d.get("data", {})
                case _:
                    return {}
        except HttpError as e:
            raise RunPodError(f"API error {e.status}: {e.body}") from e

    # =========================================================================
    # SSH Key Management (via GraphQL)
    # =========================================================================

    async def get_ssh_keys(self) -> str:
        data = await self._graphql(MYSELF_QUERY)
        return data.get("myself", {}).get("pubKey", "")

    async def ensure_ssh_key(self, public_key: str) -> None:
        existing = await self.get_ssh_keys()
        key_content = public_key.strip()

        if key_content in existing:
            self._log.debug("SSH key already registered on RunPod")
            return

        updated = f"{existing}\n{key_content}" if existing else key_content
        await self._graphql(UPDATE_USER_SETTINGS_MUTATION, {"input": {"pubKey": updated}})
        self._log.debug("SSH key registered on RunPod")

    # =========================================================================
    # Container Registry Authentication (via GraphQL)
    # =========================================================================

    async def resolve_registry_auth(self, name: str) -> str | None:
        """Resolve a container registry credential name to its ID.

        Returns None if no credential matches the given name.
        """
        data = await self._graphql(REGISTRY_AUTHS_QUERY)
        auths: list[dict[str, str]] = (
            data.get("myself", {}).get("containerRegistryCreds") or []
        )
        return next(
            (a["id"] for a in auths if a.get("name", "").lower() == name.lower()),
            None,
        )

    # =========================================================================
    # GPU Types (via GraphQL - no REST endpoint available)
    # =========================================================================

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    async def get_gpu_types(
        self,
        *,
        min_cuda_version: str | None = None,
        secure_cloud: bool | None = None,
    ) -> list[GpuTypeResponse]:
        """Get available GPU types via GraphQL API."""
        variables: dict[str, Any] = {}
        if min_cuda_version:
            variables["minCudaVersion"] = min_cuda_version
        if secure_cloud is not None:
            variables["secureCloud"] = secure_cloud
        data = await self._graphql(GPU_TYPES_QUERY, variables or None)
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
        self._log.debug("Creating instant cluster")
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
    async def create_cpu_pod(self, params: CpuPodCreateParams) -> PodResponse:
        """Create a CPU-only pod via GraphQL.

        The REST API doesn't support gpuCount=0, so we use the
        deployCpuPod GraphQL mutation instead.
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
