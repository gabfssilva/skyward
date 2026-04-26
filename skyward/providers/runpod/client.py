"""Async HTTP client for RunPod API."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import RunPod

import httpx

from skyward.infra.retry import on_status_code, retry
from skyward.observability.logger import logger

from .types import (
    ClusterCreateParams,
    ClusterResponse,
    CpuCompute,
    GpuCompute,
    GpuImage,
    GpuTypeResponse,
    NetworkVolumeResponse,
    PodDeploySpec,
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
    maxGpuCount
    maxGpuCountSecureCloud
    maxGpuCountCommunityCloud
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
      minVcpu
      minMemory
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

REGISTRY_AUTHS_QUERY = """
query { myself { containerRegistryCreds { id name } } }
"""


class RunPodError(Exception):
    """Error from RunPod API."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


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
        self._http = httpx.AsyncClient(
            base_url=RUNPOD_API_BASE,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(self._config.request_timeout),
        )

    async def __aenter__(self) -> RunPodClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self._http.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        self._log.debug(
            "HTTP {method} {path} json={json} params={params}",
            method=method, path=path, json=json, params=params,
        )
        resp = await self._http.request(method, path, json=json, params=params)
        if resp.status_code >= 400:
            self._log.warning(
                "API error {method} {url}: {status} body={body}",
                method=method, url=str(resp.request.url),
                status=resp.status_code, body=resp.text[:500],
            )
            raise RunPodError(
                f"API error {resp.status_code}: {resp.text}",
                status=resp.status_code,
            )
        return resp.json() if resp.content else None

    # =========================================================================
    # Pod Management
    # =========================================================================

    async def deploy_pod(self, spec: PodDeploySpec) -> PodResponse:
        match spec.compute, spec.placement.global_networking:
            case CpuCompute() as compute, _:
                return await self._deploy_cpu_pod(spec, compute)
            case GpuCompute() as compute, True:
                return await self._deploy_pod_via_rest(spec, compute)
            case GpuCompute() as compute, False:
                return await self._deploy_gpu_pod_via_graphql(spec, compute)

    async def _deploy_gpu_pod_via_graphql(
        self, spec: PodDeploySpec, compute: GpuCompute,
    ) -> PodResponse:
        last_error: RunPodError | None = None
        for image in compute.image_candidates:
            self._log.debug(
                "Trying pod {name}: image={image} cuda={cuda} spot={spot}",
                name=spec.name, image=image.name,
                cuda=image.allowed_cuda_versions, spot=spec.interruptible,
            )
            for country in spec.placement.country_candidates:
                try:
                    return await self._deploy_gpu_pod_once(spec, compute, image, country)
                except RunPodError as e:
                    err_str = str(e)
                    if "SUPPLY_CONSTRAINT" not in err_str and "no longer any instances available" not in err_str:
                        raise
                    self._log.info(
                        "No hosts for image={image} country={country}, trying next",
                        image=image.name, country=country,
                    )
                    last_error = e
        raise last_error or RunPodError("All image/country candidates exhausted")

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    async def _deploy_gpu_pod_once(
        self,
        spec: PodDeploySpec,
        compute: GpuCompute,
        image: GpuImage,
        country: str | None,
    ) -> PodResponse:
        input_vars: dict[str, Any] = {
            "name": spec.name,
            "imageName": image.name,
            "gpuTypeId": compute.gpu_type_id,
            "gpuCount": compute.gpu_count,
            "cloudType": spec.cloud_type,
            "containerDiskInGb": spec.storage.container_disk_gb,
            "volumeInGb": spec.storage.volume_gb,
            "volumeMountPath": spec.storage.volume_mount_path,
            "ports": ",".join(spec.ports),
            "supportPublicIp": True,
            "dockerArgs": f"bash -c '{spec.docker_args}'",
            "env": [{"key": k, "value": v} for k, v in spec.env.items()],
        }
        if spec.placement.data_center_id:
            input_vars["dataCenterId"] = spec.placement.data_center_id
        if country:
            input_vars["countryCode"] = country
        if image.allowed_cuda_versions:
            input_vars["allowedCudaVersions"] = list(image.allowed_cuda_versions)
        if spec.container_registry_auth_id:
            input_vars["containerRegistryAuthId"] = spec.container_registry_auth_id
        if spec.placement.min_download_mbps is not None:
            input_vars["minDownload"] = spec.placement.min_download_mbps
        if spec.placement.min_upload_mbps is not None:
            input_vars["minUpload"] = spec.placement.min_upload_mbps
        if spec.storage.network_volume_id:
            input_vars["networkVolumeId"] = spec.storage.network_volume_id

        if spec.interruptible:
            input_vars["bidPerGpu"] = spec.deploy_cost or compute.bid_per_gpu or 0.0
            mutation, result_key, op_name = DEPLOY_SPOT_MUTATION, "podRentInterruptable", "DeploySpot"
        else:
            if spec.deploy_cost:
                input_vars["deployCost"] = spec.deploy_cost
            mutation, result_key, op_name = DEPLOY_ON_DEMAND_MUTATION, "podFindAndDeployOnDemand", "DeployOnDemand"

        data = await self._graphql(mutation, {"input": input_vars}, op_name=op_name)
        pod: PodResponse | None = data.get(result_key)
        if not pod:
            raise RunPodError(f"Failed to deploy pod: empty response from {result_key}")
        return pod

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    async def _deploy_pod_via_rest(
        self, spec: PodDeploySpec, compute: GpuCompute,
    ) -> PodResponse:
        """REST ``POST /pods`` — sole path that accepts ``globalNetworking``.

        GraphQL ``podFindAndDeployOnDemand`` and ``podRentInterruptable`` do not
        carry the field — verified against runpod/runpodctl@v2.0.0
        (``CreatePodGQLInput``, no ``GlobalNetworking``) and runpod/runpod-python
        (``generate_pod_deployment_mutation``). Issue runpod/runpodctl#190 was
        closed but the resolution is a CLI-side validation flag, not a payload
        field.
        """
        image = compute.image_candidates[0].name
        countries = tuple(c for c in spec.placement.country_candidates if c is not None)
        params: dict[str, Any] = {
            "name": spec.name,
            "imageName": image,
            "gpuTypeIds": [compute.gpu_type_id],
            "gpuCount": compute.gpu_count,
            "cloudType": spec.cloud_type,
            "containerDiskInGb": spec.storage.container_disk_gb,
            "volumeInGb": spec.storage.volume_gb,
            "volumeMountPath": spec.storage.volume_mount_path,
            "ports": list(spec.ports),
            "dockerStartCmd": ["bash", "-c", spec.docker_args],
            "env": dict(spec.env),
            "interruptible": spec.interruptible,
            "globalNetworking": True,
            "supportPublicIp": True,
        }
        if spec.placement.data_center_id:
            params["dataCenterIds"] = [spec.placement.data_center_id]
        if countries:
            params["countryCodes"] = list(countries)
        if spec.placement.min_download_mbps is not None:
            params["minDownloadMbps"] = spec.placement.min_download_mbps
        if spec.placement.min_upload_mbps is not None:
            params["minUploadMbps"] = spec.placement.min_upload_mbps
        if spec.storage.network_volume_id:
            params["networkVolumeId"] = spec.storage.network_volume_id
        if spec.container_registry_auth_id:
            params["containerRegistryAuthId"] = spec.container_registry_auth_id

        result: PodResponse | None = await self._request("POST", "/pods", json=params)
        if not result:
            raise RunPodError("Failed to create pod: empty response")
        return result

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    async def _deploy_cpu_pod(
        self, spec: PodDeploySpec, compute: CpuCompute,
    ) -> PodResponse:
        countries = tuple(c for c in spec.placement.country_candidates if c is not None)
        params: dict[str, Any] = {
            "instanceId": compute.instance_id,
            "cloudType": spec.cloud_type,
            "containerDiskInGb": spec.storage.container_disk_gb,
            "dockerArgs": f"bash -c '{spec.docker_args}'",
            "ports": ",".join(spec.ports),
            "deployCost": spec.deploy_cost or 0.50,
            "env": [{"key": k, "value": v} for k, v in spec.env.items()],
        }
        if spec.placement.data_center_id:
            params["dataCenterId"] = spec.placement.data_center_id
        if countries:
            params["countryCode"] = countries[0]
        if spec.container_registry_auth_id:
            params["containerRegistryAuthId"] = spec.container_registry_auth_id

        data = await self._graphql(DEPLOY_CPU_POD_MUTATION, {"input": params}, op_name="DeployCpuPod")
        pod: PodResponse = data["deployCpuPod"]
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

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    async def list_network_volumes(self) -> list[NetworkVolumeResponse]:
        """List every network volume attached to this account."""
        result: list[NetworkVolumeResponse] | None = await self._request(
            "GET", "/networkvolumes",
        )
        return result or []

    @retry(
        on=lambda e: not (isinstance(e, RunPodError) and e.status == 404),
        max_attempts=10,
        base_delay=2.0,
    )
    async def terminate_pod(self, pod_id: str) -> None:
        """Terminate a pod (destroy).

        Retries until the API confirms termination (200) or the pod is
        already gone (404).  Never returns without one of those two
        confirmations.
        """
        url = f"{RUNPOD_API_BASE}/pods/{pod_id}"
        self._log.info("Terminating pod {pod_id} via DELETE {url}", pod_id=pod_id, url=url)
        try:
            result = await self._request("DELETE", f"/pods/{pod_id}")
            self._log.info(
                "Terminate pod {pod_id} response: {result}",
                pod_id=pod_id, result=result,
            )
        except RunPodError as e:
            if e.status == 404:
                self._log.info(
                    "Pod {pod_id} already terminated (404)",
                    pod_id=pod_id,
                )
                return
            self._log.error(
                "Terminate pod {pod_id} failed: status={status} body={body}",
                pod_id=pod_id, status=e.status, body=str(e)[:500],
            )
            raise

    # =========================================================================
    # GraphQL API
    # =========================================================================

    async def _graphql(
        self,
        query: str,
        variables: dict[str, Any] | None = None,
        *,
        op_name: str = "graphql",
    ) -> dict[str, Any]:
        self._log.debug(
            "GraphQL {op} variables={vars}",
            op=op_name, vars=variables,
        )
        resp = await self._http.post(
            RUNPOD_GRAPHQL_URL,
            json={"query": query, "variables": variables or {}},
            timeout=httpx.Timeout(self._config.request_timeout * 2),
        )
        if resp.status_code >= 400:
            raise RunPodError(f"API error {resp.status_code}: {resp.text}")
        data = resp.json() if resp.content else None

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

    # =========================================================================
    # Container Registry Authentication (via GraphQL)
    # =========================================================================

    async def resolve_registry_auth(self, name: str) -> str | None:
        """Resolve a container registry credential name to its ID.

        Returns None if no credential matches the given name.
        """
        data = await self._graphql(REGISTRY_AUTHS_QUERY, op_name="RegistryAuths")
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
        data = await self._graphql(GPU_TYPES_QUERY, variables or None, op_name="GpuTypes")
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
        data = await self._graphql(CREATE_CLUSTER_MUTATION, {"input": dict(params)}, op_name="CreateCluster")
        cluster: ClusterResponse = data["createCluster"]
        return cluster

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    async def delete_cluster(self, cluster_id: str) -> None:
        """Delete an Instant Cluster and all its pods.

        Args:
            cluster_id: The cluster ID to delete.
        """
        await self._graphql(DELETE_CLUSTER_MUTATION, {"input": {"clusterId": cluster_id}}, op_name="DeleteCluster")

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
