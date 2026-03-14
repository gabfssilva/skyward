"""Vultr API response types."""

from __future__ import annotations

from typing import NotRequired, TypedDict

# =============================================================================
# Shared Types
# =============================================================================


class SSHKeyResponse(TypedDict):
    """SSH key from Vultr API."""

    id: str
    name: str
    ssh_key: str
    date_created: str


# =============================================================================
# Cloud GPU (instances)
# =============================================================================


class InstanceResponse(TypedDict):
    """Cloud compute/GPU instance from Vultr API."""

    id: str
    os: str
    ram: int
    disk: int
    main_ip: str
    vcpu_count: int
    region: str
    plan: str
    status: str
    power_status: NotRequired[str]
    server_status: NotRequired[str]
    label: NotRequired[str]
    tag: NotRequired[str]
    internal_ip: NotRequired[str]
    v6_main_ip: NotRequired[str]
    os_id: NotRequired[int]
    hostname: NotRequired[str]


class InstanceCreateParams(TypedDict, total=False):
    """Parameters for creating a cloud instance."""

    region: str
    plan: str
    os_id: int
    label: str
    hostname: str
    sshkey_id: list[str]
    tag: str
    script_id: str
    enable_ipv6: bool
    user_data: str


class PlanResponse(TypedDict):
    """Cloud compute plan from Vultr API."""

    id: str
    vcpu_count: int
    ram: int
    disk: int
    disk_count: int
    bandwidth: int
    monthly_cost: float
    hourly_cost: NotRequired[float]
    type: str
    locations: list[str]
    gpu_vram_gb: NotRequired[int]
    gpu_type: NotRequired[str]


# =============================================================================
# Bare Metal
# =============================================================================


class BareMetalResponse(TypedDict):
    """Bare metal instance from Vultr API."""

    id: str
    os: str
    ram: str
    disk: str
    main_ip: str
    cpu_count: int
    region: str
    status: str
    power_status: NotRequired[str]
    plan: str
    label: NotRequired[str]
    tag: NotRequired[str]
    v6_main_ip: NotRequired[str]
    os_id: NotRequired[int]


class BareMetalCreateParams(TypedDict, total=False):
    """Parameters for creating a bare metal instance."""

    region: str
    plan: str
    os_id: int
    label: str
    hostname: str
    sshkey_id: list[str]
    tag: str
    script_id: str
    enable_ipv6: bool
    user_data: str


class MetalPlanResponse(TypedDict):
    """Bare metal plan from Vultr API."""

    id: str
    cpu_count: int
    cpu_model: str
    cpu_threads: int
    ram: int
    disk: int
    disk_count: int
    bandwidth: int
    monthly_cost: int
    type: str
    locations: list[str]
    gpu_vram_gb: NotRequired[int]
    gpu_type: NotRequired[str]


# =============================================================================
# GPU Metadata
# =============================================================================

_GPU_NAME_MAP: dict[str, str] = {
    "NVIDIA_A16": "A16",
    "NVIDIA_A40": "A40",
    "NVIDIA_A100": "A100",
    "NVIDIA_L40S": "L40S",
    "NVIDIA_HGX_H100": "H100",
    "NVIDIA_GH200": "GH200",
    "NVIDIA_HGX_B200": "B200",
    "AMD_MI300X": "MI300X",
    "AMD_MI355X": "MI355X",
}

_GPU_MEMORY_MAP: dict[str, int] = {
    "A16": 16,
    "A40": 48,
    "A100": 80,
    "L40S": 48,
    "H100": 80,
    "GH200": 80,
    "B200": 192,
    "MI300X": 192,
    "MI355X": 288,
}
