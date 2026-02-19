"""Dynamic machine type resolution for GCP Compute Engine.

Queries GCP APIs to find the best machine type for a requested
accelerator, rather than relying on static mappings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from skyward.observability.logger import logger

log = logger.bind(provider="gcp")

_ACCEL_PATTERNS: dict[str, tuple[str, ...]] = {
    "H200": ("nvidia-h200",),
    "H100": ("nvidia-h100",),
    "A100": ("nvidia-a100", "nvidia-tesla-a100"),
    "L4": ("nvidia-l4",),
    "L40": ("nvidia-l40",),
    "T4": ("nvidia-tesla-t4",),
    "V100": ("nvidia-tesla-v100",),
    "P100": ("nvidia-tesla-p100",),
    "P4": ("nvidia-tesla-p4",),
    "A10": ("nvidia-a10",),
}

_BUILTIN_GPU_FAMILIES = frozenset({"a2", "a3", "a4", "g2"})

_GUEST_ATTACHABLE = frozenset({
    "nvidia-tesla-t4",
    "nvidia-tesla-v100",
    "nvidia-tesla-p100",
    "nvidia-tesla-p4",
})

_N1_DEFAULTS: dict[int, str] = {
    1: "n1-standard-8",
    2: "n1-standard-16",
    4: "n1-standard-32",
    8: "n1-standard-96",
}


@dataclass(frozen=True, slots=True)
class ResolvedMachine:
    """Result of dynamic machine type resolution."""

    machine_type: str
    uses_guest_accelerators: bool
    accelerator_type: str
    gpu_count: int
    gpu_model: str
    gpu_vram_gb: int
    vcpus: int
    memory_gb: float


def match_accelerator_name(
    requested: str, gcp_accel_types: list[str],
) -> str:
    """Match a Skyward accelerator name to a GCP accelerator type.

    Parameters
    ----------
    requested
        Skyward accelerator name (e.g., "A100", "H100", "T4").
    gcp_accel_types
        Available GCP accelerator type names in the zone.

    Returns
    -------
    str
        Matching GCP accelerator type name.

    Raises
    ------
    RuntimeError
        If no match found.
    """
    upper = requested.upper().replace("-", "").replace("_", "")

    for name, patterns in _ACCEL_PATTERNS.items():
        if name.upper() == upper or upper in name.upper():
            for pattern in patterns:
                for gcp_type in gcp_accel_types:
                    if pattern in gcp_type:
                        return gcp_type

    normalized = upper.replace("NVIDIA", "").replace("TESLA", "").strip()
    for gcp_type in gcp_accel_types:
        gcp_norm = gcp_type.upper().replace("-", "").replace("NVIDIA", "").replace("TESLA", "")
        if normalized in gcp_norm:
            return gcp_type

    raise RuntimeError(
        f"No GCP accelerator matches '{requested}'. "
        f"Available: {', '.join(gcp_accel_types)}"
    )


def is_guest_attachable(accel_type: str) -> bool:
    """Check if accelerator type requires guest_accelerators on N1 machines."""
    return accel_type in _GUEST_ATTACHABLE


def default_n1_for_gpus(count: int) -> str:
    """Select default N1 machine type for a given GPU count."""
    return _N1_DEFAULTS.get(count, "n1-standard-8")


def select_image_family(*, has_gpu: bool) -> str:
    """Select the best image source for the instance type.

    Parameters
    ----------
    has_gpu
        Whether the instance has a GPU attached.

    Returns
    -------
    str
        Image source URI.
    """
    if has_gpu:
        return (
            "projects/deeplearning-platform-release"
            "/global/images/family/common-cu128-ubuntu-2204-nvidia-570"
        )
    return "projects/ubuntu-os-cloud/global/images/family/ubuntu-2404-lts-amd64"


def estimate_vram(accel_type: str) -> int:
    """Estimate VRAM in GB from accelerator type name."""
    match accel_type:
        case s if "h200" in s:
            return 141
        case s if "h100" in s:
            return 80
        case s if "a100" in s and "80" in s:
            return 80
        case s if "a100" in s:
            return 40
        case s if "l40" in s:
            return 48
        case s if "l4" in s:
            return 24
        case s if "v100" in s:
            return 16
        case s if "t4" in s:
            return 16
        case s if "p100" in s:
            return 16
        case s if "p4" in s:
            return 8
        case _:
            return 0


_TPU_PATTERNS = frozenset({
    "v2", "v3", "v4", "v5", "v5e", "v5p", "v5lite", "v5litepod",
    "v6", "v6e", "tpu",
})


def is_tpu_accelerator(name: str) -> bool:
    """Check if the accelerator name refers to a TPU.

    Parameters
    ----------
    name
        Skyward accelerator name (e.g., "TPUv5e", "v5litepod-4", "T4").

    Returns
    -------
    bool
        True if the name matches a TPU pattern.
    """
    normalized = name.lower().replace("_", "").replace("-", "")
    return any(normalized.startswith(p) for p in _TPU_PATTERNS)


def resolve_tpu_type(name: str) -> str:
    """Map a Skyward accelerator name to a GCP TPU accelerator type.

    Parameters
    ----------
    name
        Skyward accelerator name (e.g., "TPUv5e", "v5litepod-8", "v2-8").

    Returns
    -------
    str
        GCP TPU accelerator type (e.g., "v5litepod-8", "v2-8").
    """
    normalized = name.lower().replace("tpu", "").replace("_", "")
    match normalized:
        case s if s.startswith("v5e"):
            chips = s.removeprefix("v5e").lstrip("-") or "1"
            return f"v5litepod-{chips}"
        case s if s.startswith("v5p"):
            chips = s.removeprefix("v5p").lstrip("-") or "8"
            return f"v5p-{chips}"
        case s if s.startswith("v6e"):
            chips = s.removeprefix("v6e").lstrip("-") or "1"
            return f"v6e-{chips}"
        case s if s.startswith("v6"):
            chips = s.removeprefix("v6").lstrip("-") or "1"
            return f"v6e-{chips}"
        case s if s.startswith("v2"):
            chips = s.removeprefix("v2").lstrip("-") or "8"
            return f"v2-{chips}"
        case s if s.startswith("v3"):
            chips = s.removeprefix("v3").lstrip("-") or "8"
            return f"v3-{chips}"
        case _:
            return name.lower()


def parse_builtin_gpu_count(machine_type: str) -> int:
    """Parse GPU count from builtin GPU machine type name.

    Parameters
    ----------
    machine_type
        GCP machine type name (e.g., "a2-highgpu-4g").

    Returns
    -------
    int
        GPU count extracted from the name suffix.
    """
    match = re.search(r"-(\d+)g$", machine_type)
    return int(match.group(1)) if match else 1
