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

_BUILTIN_FAMILY_ACCELERATOR: dict[str, str] = {
    "a2": "nvidia-tesla-a100",
    "a3": "nvidia-h100-80gb",
    "g2": "nvidia-l4",
}

_KNOWN_TPU_TYPES: tuple[str, ...] = (
    "v2-8", "v2-32", "v2-128", "v2-256", "v2-512",
    "v3-8", "v3-32", "v3-128", "v3-256", "v3-512",
    "v5litepod-1", "v5litepod-4", "v5litepod-8", "v5litepod-16",
    "v5litepod-32", "v5litepod-64", "v5litepod-128", "v5litepod-256",
    "v5p-8", "v5p-16", "v5p-32", "v5p-64", "v5p-128", "v5p-256",
    "v6e-1", "v6e-4", "v6e-8", "v6e-16", "v6e-32", "v6e-64",
    "v6e-128", "v6e-256",
)

_CPU_FAMILIES = frozenset({
    "n1", "n2", "n2d", "n4", "c2", "c2d", "c3", "c3d", "c4",
    "e2", "m1", "m2", "m3", "t2d", "t2a",
})
