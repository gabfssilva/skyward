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
    return forwards.get("22", forwards.get(22, 22))


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
