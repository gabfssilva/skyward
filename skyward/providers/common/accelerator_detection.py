"""Dynamic accelerator detection via remote commands."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from skyward.types import Instance

CommandRunner = Callable[[str], str]

# ============================================================================
# NVIDIA
# ============================================================================

NVIDIA_NAME_MAP: dict[str, str] = {
    "tesla t4": "T4",
    "tesla v100": "V100",
    "tesla k80": "K80",
    "tesla p4": "P4",
    "tesla p100": "P100",
    "nvidia a2": "A2",
    "nvidia a10": "A10",
    "nvidia a10g": "A10G",
    "nvidia l4": "L4",
    "nvidia l40": "L40",
    "nvidia l40s": "L40S",
    "nvidia a100-sxm4-40gb": "A100-40",
    "nvidia a100-sxm4-80gb": "A100-80",
    "nvidia a100-pcie-40gb": "A100-40",
    "nvidia a100-pcie-80gb": "A100-80",
    "nvidia a100 80gb pcie": "A100-80",
    "nvidia h100 80gb hbm3": "H100-80GB",
    "nvidia h100 pcie": "H100-PCIe",
    "nvidia h100 sxm": "H100-SXM",
    "nvidia h100 nvl": "H100-NVL",
    "nvidia h200": "H200",
    "nvidia b100": "B100",
    "nvidia b200": "B200",
    "nvidia gh200": "GH200",
    "nvidia gb200": "GB200",
    "geforce rtx 3080": "RTX3080",
    "geforce rtx 3090": "RTX3090",
    "geforce rtx 4080": "RTX4080",
    "geforce rtx 4090": "RTX4090",
}


def _normalize_nvidia_name(raw: str) -> str:
    """Normalize nvidia-smi GPU name to accelerator type."""
    lower = raw.lower().strip()

    # Exact match
    if lower in NVIDIA_NAME_MAP:
        return NVIDIA_NAME_MAP[lower]

    # Partial match
    for pattern, normalized in NVIDIA_NAME_MAP.items():
        if pattern in lower:
            return normalized

    # Fallback: extract model (A100, H100, etc.)
    match = re.search(r"\b([a-z]?\d{2,4}[a-z]?)\b", lower, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return raw.strip()


def _detect_nvidia(run: CommandRunner) -> tuple[str | None, int]:
    """Detect NVIDIA GPUs via nvidia-smi."""
    try:
        out = run("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null")
        lines = [line.strip() for line in out.strip().split("\n") if line.strip()]
        if lines:
            return _normalize_nvidia_name(lines[0]), len(lines)
    except Exception:
        pass
    return None, 0


# ============================================================================
# AWS Neuron (Trainium / Inferentia)
# ============================================================================


def _detect_neuron(run: CommandRunner) -> tuple[str | None, int]:
    """Detect AWS Trainium/Inferentia via sysfs."""
    try:
        # Count neuron devices
        out = run(
            "ls -1d /sys/devices/virtual/neuron_device/neuron* 2>/dev/null | wc -l"
        )
        count = int(out.strip())
        if count == 0:
            return None, 0

        # Detect chip type
        model = run(
            "cat /sys/devices/virtual/neuron_device/neuron0/model 2>/dev/null "
            "|| echo unknown"
        )
        model = model.strip().lower()

        if "trn2" in model:
            return "Trainium2", count
        if "trn1" in model:
            return "Trainium1", count
        if "inf2" in model:
            return "Inferentia2", count
        if "inf1" in model:
            return "Inferentia1", count

        # Fallback
        return "Trainium1", count
    except Exception:
        pass
    return None, 0


# ============================================================================
# AMD ROCm
# ============================================================================

AMD_PATTERNS: dict[str, str] = {
    "mi50": "MI50",
    "mi100": "MI100",
    "mi210": "MI210",
    "mi250x": "MI250X",
    "mi250": "MI250",
    "mi300a": "MI300A",
    "mi300b": "MI300B",
    "mi300x": "MI300X",
}


def _detect_amd(run: CommandRunner) -> tuple[str | None, int]:
    """Detect AMD GPUs via rocm-smi."""
    try:
        out = run("rocm-smi --showproductname 2>/dev/null")
        lower = out.lower()

        for pattern, name in AMD_PATTERNS.items():
            if pattern in lower:
                # Count GPUs
                count_out = run("rocm-smi -l 2>/dev/null | grep -c 'GPU' || echo 0")
                count = max(1, int(count_out.strip()))
                return name, count
    except Exception:
        pass
    return None, 0


# ============================================================================
# Habana Gaudi
# ============================================================================


def _detect_habana(run: CommandRunner) -> tuple[str | None, int]:
    """Detect Habana Gaudi accelerators."""
    try:
        out = run("hl-smi -L 2>/dev/null")
        if "gaudi3" in out.lower():
            count = out.lower().count("hl-")
            return "Gaudi3", max(1, count)
        if "gaudi2" in out.lower():
            count = out.lower().count("hl-")
            return "Gaudi2", max(1, count)
        if "gaudi" in out.lower():
            count = out.lower().count("hl-")
            return "Gaudi", max(1, count)
    except Exception:
        pass
    return None, 0


# ============================================================================
# Public API
# ============================================================================

DETECTORS = [
    _detect_nvidia,
    _detect_neuron,
    _detect_amd,
    _detect_habana,
]


def resolve_accelerator(
    instance: Instance,
    timeout: int = 30,
) -> tuple[str | None, int]:
    """Detect accelerators by probing the instance.

    Args:
        instance: Instance to probe.
        timeout: Command timeout in seconds.

    Returns:
        Tuple of (accelerator_type, count) or (None, 0).
    """

    def run(cmd: str) -> str:
        return instance.run_command(cmd, timeout=timeout)

    for detector in DETECTORS:
        try:
            accel, count = detector(run)
            if accel and count > 0:
                return accel, count
        except Exception as err:
            print("Could not detect accelerator:", err)
            continue

    return None, 0
