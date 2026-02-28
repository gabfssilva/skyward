"""Thunder Compute API response types.

TypedDicts for API responses and GPU helper functions.
"""

from __future__ import annotations

from typing import NotRequired, TypedDict


class InstanceCreateResponse(TypedDict):
    """Instance creation response."""

    identifier: int
    uuid: str
    key: NotRequired[str]


class InstanceListItem(TypedDict):
    """Instance info from list endpoint."""

    name: str
    uuid: str
    status: str
    ip: str
    port: int
    cpuCores: int
    memory: int
    numGpus: int
    gpuType: str
    template: str
    mode: str
    storage: int
    createdAt: str


class SSHKeyResponse(TypedDict):
    """SSH key from API."""

    id: str
    name: str
    public_key: str
    fingerprint: str


class PricingEntry(TypedDict):
    """Per-GPU pricing entry."""

    prototyping: NotRequired[float]
    production: NotRequired[float]


_GPU_VRAM: dict[str, int] = {"h100": 80, "a100xl": 80, "a6000": 48}

_GPU_DISPLAY: dict[str, str] = {
    "h100": "H100",
    "a100xl": "A100 80GB",
    "a6000": "RTX A6000",
}

_GPU_TYPE_MAP: dict[str, str] = {
    "H100": "h100",
    "A100": "a100xl",
    "A100 80GB": "a100xl",
    "A6000": "a6000",
    "RTX A6000": "a6000",
}


def resolve_gpu_type(accelerator_name: str) -> str | None:
    """Map a Skyward accelerator name to a Thunder API gpu_type value.

    Parameters
    ----------
    accelerator_name
        Skyward-facing accelerator name (e.g., ``"H100"``, ``"A100 80GB"``).

    Returns
    -------
    str | None
        Thunder API gpu_type (e.g., ``"h100"``, ``"a100xl"``) or ``None``
        if the accelerator is not supported.
    """
    return _GPU_TYPE_MAP.get(accelerator_name)


def get_vram_gb(gpu_type: str) -> int:
    """Look up VRAM in GB for a Thunder GPU type.

    Parameters
    ----------
    gpu_type
        Thunder API gpu_type value (e.g., ``"h100"``, ``"a100xl"``).

    Returns
    -------
    int
        VRAM in gigabytes.

    Raises
    ------
    KeyError
        If the gpu_type is unknown.
    """
    return _GPU_VRAM[gpu_type]


def get_display_name(gpu_type: str) -> str:
    """Look up the display name for a Thunder GPU type.

    Parameters
    ----------
    gpu_type
        Thunder API gpu_type value (e.g., ``"h100"``, ``"a100xl"``).

    Returns
    -------
    str
        Human-readable display name (e.g., ``"H100"``, ``"A100 80GB"``).

    Raises
    ------
    KeyError
        If the gpu_type is unknown.
    """
    return _GPU_DISPLAY[gpu_type]
