"""TensorDock API response types and helpers."""

from __future__ import annotations

from typing import NotRequired, TypedDict

# =============================================================================
# Hostnode types (from /api/v0/client/deploy/hostnodes)
# =============================================================================


class HostnodeGpu(TypedDict):
    amount: int


class HostnodeSpecs(TypedDict):
    gpu: dict[str, HostnodeGpu]
    ram: dict[str, int]
    cpu: dict[str, int]
    storage: dict[str, int]


class HostnodePricing(TypedDict):
    gpu: dict[str, float]
    ram: float
    cpu: float
    storage: float


class HostnodeLocation(TypedDict):
    country: str
    city: str
    region: str


class HostnodeNetworking(TypedDict):
    ports: NotRequired[int]
    receive: NotRequired[float]
    send: NotRequired[float]


class HostnodeResponse(TypedDict):
    specs: HostnodeSpecs
    networking: NotRequired[HostnodeNetworking]
    location: HostnodeLocation
    pricing: NotRequired[HostnodePricing]


# =============================================================================
# Deploy types (from /api/v0/client/deploy/single)
# =============================================================================


class DeployResponse(TypedDict):
    success: bool
    ip: NotRequired[str]
    port_forwards: NotRequired[dict[str, int]]
    server: NotRequired[str]
    error: NotRequired[str]


# =============================================================================
# VM types (from /api/v0/client/get/single, /api/v0/client/list)
# =============================================================================


class CostInfo(TypedDict):
    on: NotRequired[float]
    off: NotRequired[float]


class VmDetails(TypedDict):
    ip: NotRequired[str]
    status: NotRequired[str]
    port_forwards: NotRequired[dict[str, int]]
    gpu_model: NotRequired[str]
    gpu_quantity: NotRequired[int]
    vcpus: NotRequired[int]
    ram: NotRequired[int]
    storage: NotRequired[int]
    cost_per_hour: NotRequired[CostInfo]
    name: NotRequired[str]
    operating_system: NotRequired[str]


class VmGetResponse(TypedDict):
    success: bool
    virtualmachine: NotRequired[VmDetails]


class VmListResponse(TypedDict):
    success: bool
    virtualmachines: NotRequired[dict[str, VmDetails]]


class BillingResponse(TypedDict):
    success: bool
    balance: NotRequired[float]
    spent: NotRequired[float]


class AuthTestResponse(TypedDict):
    success: bool


# =============================================================================
# GPU mapping and helpers
# =============================================================================


_GPU_NAME_MAP: dict[str, str] = {
    "h100-sxm5-80gb": "H100",
    "a100-sxm4-80gb": "A100",
    "a100-pcie-80gb": "A100",
    "a6000-pcie-48gb": "A6000",
    "l40-pcie-48gb": "L40",
    "l40s-pcie-48gb": "L40S",
    "rtxa6000-pcie-48gb": "RTX A6000",
    "rtxa5000-pcie-24gb": "RTX A5000",
    "rtxa4000-pcie-16gb": "RTX A4000",
    "geforcertx4090-pcie-24gb": "RTX 4090",
    "geforcertx4080-pcie-16gb": "RTX 4080",
    "geforcertx3090-pcie-24gb": "RTX 3090",
    "geforcertx3080-pcie-12gb": "RTX 3080",
    "geforcertx3080-pcie-10gb": "RTX 3080",
    "v100-sxm2-16gb": "V100",
    "v100-pcie-16gb": "V100",
}

_GPU_MEMORY_MAP: dict[str, int] = {
    "h100-sxm5-80gb": 80,
    "a100-sxm4-80gb": 80,
    "a100-pcie-80gb": 80,
    "a6000-pcie-48gb": 48,
    "l40-pcie-48gb": 48,
    "l40s-pcie-48gb": 48,
    "rtxa6000-pcie-48gb": 48,
    "rtxa5000-pcie-24gb": 24,
    "rtxa4000-pcie-16gb": 16,
    "geforcertx4090-pcie-24gb": 24,
    "geforcertx4080-pcie-16gb": 16,
    "geforcertx3090-pcie-24gb": 24,
    "geforcertx3080-pcie-12gb": 12,
    "geforcertx3080-pcie-10gb": 10,
    "v100-sxm2-16gb": 16,
    "v100-pcie-16gb": 16,
}


def normalize_gpu_name(td_gpu_id: str) -> str:
    """Map TensorDock GPU ID to display name.

    Examples
    --------
    >>> normalize_gpu_name("geforcertx4090-pcie-24gb")
    'RTX 4090'
    >>> normalize_gpu_name("unknown-gpu")
    'unknown-gpu'
    """
    return _GPU_NAME_MAP.get(td_gpu_id, td_gpu_id)


def get_gpu_memory_gb(td_gpu_id: str) -> int:
    """Extract GPU memory in GB from TensorDock GPU ID.

    Falls back to parsing the suffix (e.g., '-24gb' -> 24).

    Examples
    --------
    >>> get_gpu_memory_gb("geforcertx4090-pcie-24gb")
    24
    """
    if mem := _GPU_MEMORY_MAP.get(td_gpu_id):
        return mem
    parts = td_gpu_id.rsplit("-", 1)
    if len(parts) == 2 and parts[1].endswith("gb"):
        try:
            return int(parts[1][:-2])
        except ValueError:
            pass
    return 0


def get_ssh_port(vm: VmDetails) -> int:
    """Extract external SSH port from port_forwards mapping.

    TensorDock maps internal -> external ports. SSH (internal 22)
    is mapped to a random external port.

    Examples
    --------
    >>> get_ssh_port({"port_forwards": {"22": 34567}})
    34567
    """
    forwards = vm.get("port_forwards") or {}
    return forwards.get("22", 22)


def _normalize_string(s: str) -> str:
    """Normalize string for GPU name comparison."""
    return s.upper().replace("_", "").replace("-", "").replace(" ", "")


def gpu_matches(td_gpu_id: str, requested: str) -> bool:
    """Check if a TensorDock GPU ID matches a requested GPU name.

    Fuzzy match: compares normalized display names, ignoring case
    and underscores/hyphens/spaces.

    Examples
    --------
    >>> gpu_matches("geforcertx4090-pcie-24gb", "RTX 4090")
    True
    >>> gpu_matches("geforcertx4090-pcie-24gb", "RTX_4090")
    True
    """
    display = normalize_gpu_name(td_gpu_id)
    display_norm = _normalize_string(display)
    requested_norm = _normalize_string(requested)
    td_norm = _normalize_string(td_gpu_id)
    return display_norm == requested_norm or td_norm == requested_norm


# =============================================================================
# v2 Location types (from GET /api/v2/locations)
# =============================================================================


class LocationGpuPricing(TypedDict):
    per_vcpu_hr: float
    per_gb_ram_hr: float
    per_gb_storage_hr: float


class LocationGpuNetworkFeatures(TypedDict):
    dedicated_ip_available: bool
    port_forwarding_available: bool
    network_storage_available: bool


class LocationGpuResources(TypedDict):
    max_vcpus: int
    max_ram_gb: int
    max_storage_gb: int


class LocationGpu(TypedDict):
    v0Name: str
    displayName: str
    max_count: int
    price_per_hr: float
    resources: LocationGpuResources
    pricing: LocationGpuPricing
    network_features: LocationGpuNetworkFeatures


class LocationsData(TypedDict):
    locations: list[Location]


class Location(TypedDict):
    id: str
    city: str
    stateprovince: str
    country: str
    tier: int
    gpus: list[LocationGpu]


class LocationsResponse(TypedDict):
    data: LocationsData


# =============================================================================
# v2 Instance types (from /api/v2/instances)
# =============================================================================


class PortForward(TypedDict):
    internal_port: int
    external_port: int


class InstanceGpuInfo(TypedDict):
    count: int
    v0Name: NotRequired[str]


class InstanceResources(TypedDict):
    vcpu_count: int
    ram_gb: int
    storage_gb: int
    gpus: dict[str, InstanceGpuInfo]


class V2InstanceResponse(TypedDict):
    type: str
    id: str
    name: NotRequired[str]
    status: NotRequired[str]
    ipAddress: NotRequired[str]
    portForwards: NotRequired[list[PortForward]]
    resources: NotRequired[InstanceResources]
    rateHourly: NotRequired[float]


class V2InstanceCreateResponse(TypedDict):
    data: V2InstanceResponse


class V2InstanceListData(TypedDict):
    instances: list[V2InstanceResponse]


class V2InstanceListResponse(TypedDict):
    data: V2InstanceListData


class V2AuthTestResponse(TypedDict):
    success: bool
    organizationId: NotRequired[str]


# =============================================================================
# v2 helpers
# =============================================================================


def normalize_gpu_name_v2(gpu: LocationGpu) -> str:
    """Extract display name from v2 LocationGpu.

    Uses the displayName field directly from the API response,
    falling back to v0Name if displayName is empty.

    """
    return gpu.get("displayName") or gpu.get("v0Name", "")


def get_gpu_memory_gb_v2(gpu: LocationGpu) -> int:
    """Extract GPU memory in GB from v2 LocationGpu.

    Parses the v0Name suffix (e.g., 'geforcertx5090-pcie-32gb' -> 32).
    """
    return get_gpu_memory_gb(gpu.get("v0Name", ""))


def gpu_matches_v2(gpu: LocationGpu, requested: str) -> bool:
    """Check if a v2 LocationGpu matches a requested GPU name.

    Checks against the static name map (v0Name → short name), the full
    displayName (contains check), and the raw v0Name, all normalized.
    """
    requested_norm = _normalize_string(requested)
    # Static map: "geforcertx4090-pcie-24gb" → "RTX 4090"
    mapped_norm = _normalize_string(normalize_gpu_name(gpu.get("v0Name", "")))
    if mapped_norm == requested_norm:
        return True
    # Contains: "NVIDIA GeForce RTX 4090 PCIe 24GB" contains "RTX 4090"
    display_norm = _normalize_string(gpu.get("displayName", ""))
    if requested_norm in display_norm:
        return True
    v0_norm = _normalize_string(gpu.get("v0Name", ""))
    return requested_norm in v0_norm


def get_ssh_port_v2(port_forwards: list[PortForward]) -> int:
    """Extract external SSH port from v2 portForwards array.

    Examples
    --------
    >>> get_ssh_port_v2([{"internal_port": 22, "external_port": 34567}])
    34567
    >>> get_ssh_port_v2([])
    22
    """
    for pf in port_forwards:
        if pf.get("internal_port") == 22:
            return pf.get("external_port", 22)
    return 22


_V2_IMAGE_MAP: dict[str, str] = {
    "Ubuntu 22.04 LTS": "ubuntu2204",
    "Ubuntu 24.04 LTS": "ubuntu2404",
}

_GPU_DEFAULT_IMAGE = "ubuntu2404_ml_everything"


def resolve_v2_image(operating_system: str, *, has_gpu: bool = False) -> str:
    """Map legacy OS name to v2 image ID, or pass through if already v2 format.

    Examples
    --------
    >>> resolve_v2_image("Ubuntu 22.04 LTS")
    'ubuntu2204'
    >>> resolve_v2_image("ubuntu2404")
    'ubuntu2404'
    >>> resolve_v2_image("ubuntu2404", has_gpu=True)
    'ubuntu2404_ml_everything'
    """
    image = _V2_IMAGE_MAP.get(operating_system, operating_system)
    if has_gpu and image == "ubuntu2404":
        return _GPU_DEFAULT_IMAGE
    return image
