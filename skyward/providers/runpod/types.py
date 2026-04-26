"""RunPod API response types and deploy spec."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict

# =============================================================================
# Response Types
# =============================================================================


class GpuInfo(TypedDict):
    """GPU configuration from pod response."""

    count: int
    id: NotRequired[str]
    gpuType: NotRequired[str]


class MachineInfo(TypedDict):
    """Machine/host info from pod response."""

    # Location
    location: NotRequired[str]
    dataCenterId: NotRequired[str]
    secureCloud: NotRequired[bool]

    # GPU
    gpuDisplayName: NotRequired[str]
    gpuTypeId: NotRequired[str]
    gpuAvailable: NotRequired[int]
    currentPricePerGpu: NotRequired[float]
    minPodGpuCount: NotRequired[int]

    # CPU
    cpuCount: NotRequired[int]
    cpuTypeId: NotRequired[str]

    # Network
    maxDownloadSpeedMbps: NotRequired[int]
    maxUploadSpeedMbps: NotRequired[int]
    diskThroughputMBps: NotRequired[int]
    supportPublicIp: NotRequired[bool]

    # Maintenance
    maintenanceStart: NotRequired[str]
    maintenanceEnd: NotRequired[str]
    maintenanceNote: NotRequired[str]
    note: NotRequired[str]

    # Pricing
    costPerHr: NotRequired[float]


class PortMapping(TypedDict):
    """Port mapping from pod response."""

    internalPort: int
    externalPort: int
    protocol: str


class PodResponse(TypedDict):
    """Pod response from RunPod API."""

    id: str
    name: NotRequired[str]
    desiredStatus: str  # RUNNING, EXITED, TERMINATED
    publicIp: NotRequired[str | None]
    costPerHr: NotRequired[float]
    adjustedCostPerHr: NotRequired[float]
    machine: NotRequired[MachineInfo]
    gpu: NotRequired[GpuInfo]
    gpuCount: NotRequired[int]
    vcpuCount: NotRequired[int]
    memoryInGb: NotRequired[float]
    portMappings: NotRequired[dict[str, int] | None]  # {"22": 22008}
    imageName: NotRequired[str]
    interruptible: NotRequired[bool]
    # REST /pods/{id} returns these at the top level in addition to the
    # nested ``machine`` block; keep both as fallback sources.
    dataCenterId: NotRequired[str]
    location: NotRequired[str]


class PodCreateParams(TypedDict, total=False):
    """Parameters for creating a pod via REST API."""

    name: str
    imageName: str
    gpuTypeIds: list[str]
    gpuCount: int
    cloudType: str  # SECURE, COMMUNITY
    containerDiskInGb: int
    volumeInGb: int
    volumeMountPath: str
    ports: list[str]
    dockerStartCmd: list[str]
    env: dict[str, str]
    interruptible: bool
    dataCenterIds: list[str]
    countryCodes: list[str]
    minVCPUPerGPU: int
    minRAMPerGPU: int
    minDownloadMbps: int
    minUploadMbps: int
    globalNetworking: bool
    supportPublicIp: bool
    networkVolumeId: NotRequired[str]


class NetworkVolumeResponse(TypedDict):
    """RunPod network volume resource (``GET /v1/networkvolumes``)."""

    id: str
    name: NotRequired[str]
    dataCenterId: NotRequired[str]
    size: NotRequired[int]


class LowestPriceResponse(TypedDict):
    """Lowest price + availability from RunPod API."""

    minimumBidPrice: NotRequired[float]
    uninterruptablePrice: NotRequired[float]
    stockStatus: NotRequired[str | None]
    totalCount: NotRequired[int]
    rentedCount: NotRequired[int]
    minVcpu: NotRequired[int | None]
    minMemory: NotRequired[int | None]


class GpuTypeResponse(TypedDict):
    """GPU type from RunPod API."""

    id: str
    displayName: str
    memoryInGb: int
    secureCloud: bool
    communityCloud: bool
    maxGpuCount: NotRequired[int]
    maxGpuCountSecureCloud: NotRequired[int]
    maxGpuCountCommunityCloud: NotRequired[int]
    lowestPrice: NotRequired[LowestPriceResponse]


# =============================================================================
# Instant Cluster Types
# =============================================================================


class ClusterPodInfo(TypedDict):
    """Pod info from cluster response."""

    id: str
    desiredStatus: str
    publicIp: NotRequired[str | None]
    clusterIp: NotRequired[str | None]  # Internal IP for inter-node communication
    clusterIdx: NotRequired[int]  # Index within cluster (0, 1, 2...)


class ClusterResponse(TypedDict):
    """Cluster response from RunPod GraphQL API."""

    id: str
    name: str
    pods: list[ClusterPodInfo]


class ClusterCreateParams(TypedDict, total=False):
    """Parameters for creating an instant cluster."""

    clusterName: str
    gpuTypeId: str  # Required
    podCount: int  # Required
    gpuCountPerPod: int  # Required
    type: str  # Required: TRAINING, APPLICATION, SLURM
    imageName: str
    dockerArgs: str
    containerDiskInGb: int
    volumeInGb: int
    volumeMountPath: str
    ports: str  # Comma-separated, e.g. "22/tcp,8265/http"
    dataCenterId: str
    countryCode: str
    env: list[dict[str, str]]  # [{"key": "...", "value": "..."}]
    networkVolumeId: str
    templateId: str
    deployCost: float
    throughput: int
    containerRegistryAuthId: str


class CpuPodCreateParams(TypedDict, total=False):
    """Parameters for creating a CPU-only pod via GraphQL."""

    instanceId: str  # e.g. "cpu3c-2-4" (3 cores, 2GB RAM, 4GB disk)
    cloudType: str  # SECURE or COMMUNITY
    containerDiskInGb: int
    deployCost: float
    dataCenterId: str | None
    countryCode: str | None
    networkVolumeId: str | None
    startJupyter: bool
    ports: str  # e.g. "22/tcp"
    dockerArgs: str
    volumeKey: str | None
    env: list[dict[str, str]]  # [{"key": "...", "value": "..."}]
    containerRegistryAuthId: str


# =============================================================================
# Helper Functions
# =============================================================================


def get_ssh_port(pod: PodResponse) -> int:
    """Extract SSH port from pod port mappings.

    RunPod returns portMappings as {"internal_port": external_port}, e.g. {"22": 22008}
    """
    mappings = pod.get("portMappings") or {}
    return mappings.get("22", 22)


def get_gpu_model(pod: PodResponse) -> str:
    """Extract GPU model from pod response."""
    machine = pod.get("machine") or {}
    if name := machine.get("gpuDisplayName") or machine.get("gpuTypeId"):
        return name
    gpu = pod.get("gpu")
    if gpu:
        return gpu.get("displayName") or gpu.get("gpuType") or ""
    return ""


type CloudType = Literal["SECURE", "COMMUNITY"]


@dataclass(frozen=True, slots=True)
class GpuImage:
    name: str
    allowed_cuda_versions: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class GpuCompute:
    gpu_type_id: str
    gpu_count: int
    image_candidates: tuple[GpuImage, ...]
    bid_per_gpu: float | None = None


@dataclass(frozen=True, slots=True)
class CpuCompute:
    instance_id: str


type Compute = GpuCompute | CpuCompute


@dataclass(frozen=True, slots=True)
class Storage:
    container_disk_gb: int
    volume_gb: int
    volume_mount_path: str
    network_volume_id: str | None = None


@dataclass(frozen=True, slots=True)
class Placement:
    data_center_id: str | None = None
    country_candidates: tuple[str | None, ...] = (None,)
    global_networking: bool = False
    min_download_mbps: int | None = None
    min_upload_mbps: int | None = None


@dataclass(frozen=True, slots=True)
class PodDeploySpec:
    name: str
    cloud_type: CloudType
    compute: Compute
    storage: Storage
    placement: Placement
    ports: tuple[str, ...]
    docker_args: str
    env: Mapping[str, str]
    interruptible: bool = False
    deploy_cost: float | None = None
    container_registry_auth_id: str | None = None
