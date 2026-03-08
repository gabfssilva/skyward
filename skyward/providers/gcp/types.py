"""GCP provider types and constants.

Dataclasses, type definitions, and constant mappings for GCP Compute Engine.
"""

from __future__ import annotations

from dataclasses import dataclass


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


@dataclass(frozen=True, slots=True)
class GCPSpecific:
    """GCP-specific cluster data flowing through Cluster[GCPSpecific]."""

    project: str
    zone: str
    template_name: str
    firewall_rule: str | None
    machine_type: str
    image: str
    uses_guest_accelerators: bool
    accelerator_type: str
    gpu_count: int = 0
    gpu_model: str = ""
    vcpus: int = 0
    memory_gb: float = 0.0
    gpu_vram_gb: int = 0


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

_TPU_PATTERNS = frozenset({
    "v2", "v3", "v4", "v5", "v5e", "v5p", "v5lite", "v5litepod",
    "v6", "v6e", "tpu",
})
