"""RunPod API response types.

TypedDicts for API responses - no conversion needed.
"""

from __future__ import annotations

from typing import NotRequired, TypedDict

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
    env: dict[str, str]
    interruptible: bool
    dataCenterIds: list[str]
    minVCPUPerGPU: int
    minRAMPerGPU: int
    globalNetworking: bool


class GpuTypeResponse(TypedDict):
    """GPU type from RunPod API."""

    id: str
    displayName: str
    memoryInGb: int
    secureCloud: bool
    communityCloud: bool
    lowestPrice: NotRequired[dict[str, float]]


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
    startSsh: bool
    containerDiskInGb: int
    volumeInGb: int
    volumeMountPath: str
    ports: str  # Comma-separated, e.g. "22/tcp,8265/http"
    dataCenterId: str
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
    networkVolumeId: str | None
    startJupyter: bool
    startSsh: bool
    templateId: str  # e.g. "runpod-ubuntu"
    ports: str  # e.g. "22/tcp"
    dockerArgs: str  # Custom docker args
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
