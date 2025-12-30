"""Device partitioning strategies for different accelerator types.

Provides a registry-based system for creating device partitioning strategies
that handle multi-GPU, single-GPU, and MIG configurations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from skyward.accelerator import (
    MIG_CAPABLE,
    MIG_MAX_INSTANCES,
    MIG_PROFILE_SLICES,
    _AMD_VALUES,
    _HABANA_VALUES,
    _INFERENTIA_VALUES,
    _NVIDIA_VALUES,
    _TRAINIUM_VALUES,
    _normalize_accelerator,
)


# Type alias for environment function
# (worker_id, device_count, workers_per_gpu) -> env vars
EnvFn = Callable[[int, int, int], dict[str, str]]


@dataclass(frozen=True, slots=True)
class PartitionStrategy:
    """Strategy for device assignment to workers.

    Attributes:
        workers_per_gpu: Workers per GPU (1 = single GPU, >1 = MIG)
        setup_script: Shell commands for bootstrap (MIG setup, etc.)
        env_fn: Function to generate env vars per worker
        mig_profiles: MIG profiles per worker (None if no MIG)
    """

    workers_per_gpu: int
    setup_script: str
    env_fn: EnvFn
    mig_profiles: tuple[str, ...] | None = None

    def get_worker_env(
        self,
        worker_id: int,
        device_count: int,
    ) -> dict[str, str]:
        """Get environment variables for a specific worker.

        Args:
            worker_id: Worker ID (0-indexed).
            device_count: Total device count on instance.

        Returns:
            Environment variables for device assignment.
        """
        return self.env_fn(worker_id, device_count, self.workers_per_gpu)


# =============================================================================
# NVIDIA Handlers
# =============================================================================


def _nvidia_multi_gpu_env(
    worker_id: int,
    device_count: int,
    gpus_per_worker: int,
) -> dict[str, str]:
    """Assign multiple GPUs to a worker."""
    start = worker_id * gpus_per_worker
    if start + gpus_per_worker > device_count:
        # Wrap around if needed
        devices = [str((start + i) % device_count) for i in range(gpus_per_worker)]
    else:
        devices = [str(start + i) for i in range(gpus_per_worker)]
    return {"CUDA_VISIBLE_DEVICES": ",".join(devices)}


def _nvidia_single_gpu_env(
    worker_id: int,
    device_count: int,
    workers_per_gpu: int,
) -> dict[str, str]:
    """Assign single GPU to a worker."""
    gpu_id = worker_id % device_count
    return {"CUDA_VISIBLE_DEVICES": str(gpu_id)}


def _nvidia_mig_env(
    worker_id: int,
    device_count: int,
    workers_per_gpu: int,
) -> dict[str, str]:
    """Assign MIG partition to a worker.

    MIG UUIDs are resolved by the bootstrap script after MIG setup.
    """
    gpu_id = worker_id // workers_per_gpu
    partition_id = worker_id % workers_per_gpu

    # Placeholder that will be resolved by MIG setup script
    return {
        "CUDA_VISIBLE_DEVICES": f"${{MIG_UUID_{gpu_id}_{partition_id}}}",
        "SKYWARD_MIG_GPU": str(gpu_id),
        "SKYWARD_MIG_PARTITION": str(partition_id),
    }


def _generate_mig_setup_script(
    profiles: tuple[str, ...],
    device_count: int,
    worker_count: int,
) -> str:
    """Generate shell script to setup MIG on NVIDIA GPUs.

    Supports both homogeneous (same profile) and heterogeneous (different profiles)
    MIG configurations.

    Args:
        profiles: MIG profiles per partition (e.g., ("3g.40gb", "3g.40gb") or
                  ("4g.40gb", "3g.40gb") for heterogeneous).
        device_count: Number of GPUs on instance.
        worker_count: Total number of workers.

    Returns:
        Shell script fragment for bootstrap.
    """
    workers_per_gpu = len(profiles)

    # Generate profile creation commands
    profile_commands = "\n".join(
        f'    nvidia-smi mig -i $gpu -cgi {profile} -C'
        for profile in profiles
    )

    return f"""
# Enable MIG mode on all GPUs
for i in $(seq 0 $(({device_count} - 1))); do
    nvidia-smi -i $i -mig 1 || true
done

# Wait for MIG mode to settle
sleep 2

# Create MIG instances
for gpu in $(seq 0 $(({device_count} - 1))); do
    # Destroy existing MIG instances
    nvidia-smi mig -i $gpu -dci || true
    nvidia-smi mig -i $gpu -dgi || true

    # Create GPU instances with profiles
{profile_commands}
done

# Wait for MIG instances to be ready
sleep 1

# Extract all MIG UUIDs to a temp file
# Format: "  MIG 3g.40gb     Device  0: (UUID: MIG-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)"
MIG_UUID_FILE=/tmp/mig_uuids.txt
nvidia-smi -L 2>/dev/null | sed -n 's/.*\\(MIG-[a-f0-9-]*\\).*/\\1/p' > "$MIG_UUID_FILE"

echo "Found MIG UUIDs:"
cat "$MIG_UUID_FILE"

# Read UUIDs into array (avoiding subshell issues)
declare -a ALL_UUIDS
mapfile -t ALL_UUIDS < "$MIG_UUID_FILE"

# Resolve UUID placeholders in worker env files
# Workers are numbered 0..N-1, MIG devices are in order from nvidia-smi -L
for worker in $(seq 0 $(({worker_count} - 1))); do
    gpu=$((worker / {workers_per_gpu}))
    partition=$((worker % {workers_per_gpu}))
    # UUID index = gpu * workers_per_gpu + partition
    uuid_idx=$((gpu * {workers_per_gpu} + partition))
    uuid="${{ALL_UUIDS[$uuid_idx]:-}}"

    if [ -f /opt/skyward/worker-$worker.env ]; then
        if [ -n "$uuid" ]; then
            # Replace placeholder with actual MIG UUID
            sed -i "s|\\${{MIG_UUID_${{gpu}}_${{partition}}}}|$uuid|g" /opt/skyward/worker-$worker.env
            echo "Resolved worker-$worker.env: MIG_UUID_${{gpu}}_${{partition}} -> $uuid"
        else
            echo "WARNING: No MIG UUID found for worker $worker (GPU $gpu partition $partition, index $uuid_idx)"
        fi
    fi
done

rm -f "$MIG_UUID_FILE"
"""


def _nvidia_partition(
    accelerator: str,
    gpus_per_worker: int,
    mig: str | tuple[str, ...] | None,
    device_count: int,
) -> PartitionStrategy:
    """Create partition strategy for NVIDIA GPUs.

    Args:
        accelerator: GPU type (e.g., "H100-80GB").
        gpus_per_worker: GPUs per worker (for multi-GPU). Ignored if mig is set.
        mig: MIG profile(s) or None.
        device_count: Total GPUs on instance.

    Returns:
        PartitionStrategy for this configuration.
    """
    if mig is not None:
        # MIG partitioning
        normalized = _normalize_accelerator(accelerator)
        if normalized not in MIG_CAPABLE:
            raise ValueError(
                f"{accelerator} does not support MIG partitioning"
            )

        # Normalize to tuple
        profiles = (mig,) * MIG_MAX_INSTANCES[mig] if isinstance(mig, str) else mig
        workers_per_gpu = len(profiles)
        worker_count = device_count * workers_per_gpu

        return PartitionStrategy(
            workers_per_gpu=workers_per_gpu,
            setup_script=_generate_mig_setup_script(profiles, device_count, worker_count),
            env_fn=_nvidia_mig_env,
            mig_profiles=profiles,
        )
    elif gpus_per_worker > 1:
        # Multi-GPU per worker
        return PartitionStrategy(
            workers_per_gpu=1,  # Actually gpus_per_worker GPUs for 1 worker
            setup_script="",
            env_fn=lambda w, d, _: _nvidia_multi_gpu_env(w, d, gpus_per_worker),
        )
    else:
        # Single GPU per worker
        return PartitionStrategy(
            workers_per_gpu=1,
            setup_script="",
            env_fn=_nvidia_single_gpu_env,
        )


# =============================================================================
# AWS Trainium/Inferentia Handlers
# =============================================================================


# Neuron cores per device type
NEURON_CORES: Final[dict[str, int]] = {
    "Trainium1": 2,   # trn1: 2 cores per chip
    "Trainium2": 2,   # trn2: 2 cores per chip
    "Trainium3": 2,   # trn3: estimated
    "Inferentia1": 4, # inf1: 4 cores per chip
    "Inferentia2": 2, # inf2: 2 cores per chip
}


def _trainium_env(
    worker_id: int,
    device_count: int,
    workers_per_gpu: int,
) -> dict[str, str]:
    """Assign Neuron cores to a worker."""
    if workers_per_gpu == 1:
        # Full device per worker
        return {"NEURON_RT_VISIBLE_CORES": str(worker_id % device_count)}
    else:
        # Shared cores (not typically used for Trainium)
        cores_per_device = 2
        total_cores = device_count * cores_per_device
        cores_per_worker = total_cores // (device_count * workers_per_gpu)
        start_core = worker_id * cores_per_worker
        end_core = start_core + cores_per_worker - 1
        return {
            "NEURON_RT_VISIBLE_CORES": f"{start_core}-{end_core}",
            "NEURON_RT_NUM_CORES": str(cores_per_worker),
        }


def _trainium_partition(
    accelerator: str,
    gpus_per_worker: int,
    mig: str | tuple[str, ...] | None,
    device_count: int,
) -> PartitionStrategy:
    """Create partition strategy for AWS Trainium/Inferentia."""
    return PartitionStrategy(
        workers_per_gpu=1,
        setup_script="",
        env_fn=_trainium_env,
    )


# =============================================================================
# AMD GPU Handlers
# =============================================================================


def _amd_env(
    worker_id: int,
    device_count: int,
    workers_per_gpu: int,
) -> dict[str, str]:
    """Assign AMD GPUs to a worker."""
    gpu_id = worker_id % device_count
    return {"HIP_VISIBLE_DEVICES": str(gpu_id)}


def _amd_multi_gpu_env(
    worker_id: int,
    device_count: int,
    gpus_per_worker: int,
) -> dict[str, str]:
    """Assign multiple AMD GPUs to a worker."""
    start = worker_id * gpus_per_worker
    devices = ",".join(str((start + i) % device_count) for i in range(gpus_per_worker))
    return {"HIP_VISIBLE_DEVICES": devices}


def _amd_partition(
    accelerator: str,
    gpus_per_worker: int,
    mig: str | tuple[str, ...] | None,
    device_count: int,
) -> PartitionStrategy:
    """Create partition strategy for AMD GPUs."""
    if mig is not None:
        raise ValueError("AMD GPUs do not support MIG partitioning")

    if gpus_per_worker > 1:
        return PartitionStrategy(
            workers_per_gpu=1,
            setup_script="",
            env_fn=lambda w, d, _: _amd_multi_gpu_env(w, d, gpus_per_worker),
        )

    return PartitionStrategy(
        workers_per_gpu=1,
        setup_script="",
        env_fn=_amd_env,
    )


# =============================================================================
# Habana Gaudi Handlers
# =============================================================================


def _habana_env(
    worker_id: int,
    device_count: int,
    workers_per_gpu: int,
) -> dict[str, str]:
    """Assign Habana Gaudi devices to a worker."""
    device_id = worker_id % device_count
    return {"HABANA_VISIBLE_DEVICES": str(device_id)}


def _habana_multi_device_env(
    worker_id: int,
    device_count: int,
    devices_per_worker: int,
) -> dict[str, str]:
    """Assign multiple Habana devices to a worker."""
    start = worker_id * devices_per_worker
    devices = ",".join(str((start + i) % device_count) for i in range(devices_per_worker))
    return {"HABANA_VISIBLE_DEVICES": devices}


def _habana_partition(
    accelerator: str,
    gpus_per_worker: int,
    mig: str | tuple[str, ...] | None,
    device_count: int,
) -> PartitionStrategy:
    """Create partition strategy for Habana Gaudi."""
    if mig is not None:
        raise ValueError("Habana Gaudi does not support MIG partitioning")

    if gpus_per_worker > 1:
        return PartitionStrategy(
            workers_per_gpu=1,
            setup_script="",
            env_fn=lambda w, d, _: _habana_multi_device_env(w, d, gpus_per_worker),
        )

    return PartitionStrategy(
        workers_per_gpu=1,
        setup_script="",
        env_fn=_habana_env,
    )


# =============================================================================
# CPU-only Handler
# =============================================================================


def _cpu_only_env(
    worker_id: int,
    device_count: int,
    workers_per_gpu: int,
) -> dict[str, str]:
    """No device env vars for CPU-only."""
    return {}


def _cpu_only_partition(
    accelerator: str,
    gpus_per_worker: int,
    mig: str | tuple[str, ...] | None,
    device_count: int,
) -> PartitionStrategy:
    """Create partition strategy for CPU-only workloads."""
    return PartitionStrategy(
        workers_per_gpu=1,
        setup_script="",
        env_fn=_cpu_only_env,
    )


# =============================================================================
# Registry
# =============================================================================


def _get_accelerator_family(accelerator: str) -> str:
    """Determine accelerator family from type string."""
    # Normalize common aliases
    normalized = accelerator.split("-")[0].upper()

    # Check against known values
    if accelerator in _NVIDIA_VALUES or normalized in {"H100", "A100", "T4", "V100", "L4", "L40", "A10", "RTX"}:
        return "nvidia"
    if accelerator in _TRAINIUM_VALUES or normalized in {"TRAINIUM", "TRN"}:
        return "trainium"
    if accelerator in _INFERENTIA_VALUES or normalized in {"INFERENTIA", "INF"}:
        return "trainium"  # Same handler
    if accelerator in _AMD_VALUES or normalized in {"MI"}:
        return "amd"
    if accelerator in _HABANA_VALUES or normalized in {"GAUDI"}:
        return "habana"

    return "unknown"


# Handler registry - new signature with mig parameter
PartitionHandler = Callable[
    [str, int, str | tuple[str, ...] | None, int],
    PartitionStrategy,
]

_PARTITION_HANDLERS: Final[dict[str, PartitionHandler]] = {
    "nvidia": _nvidia_partition,
    "trainium": _trainium_partition,
    "amd": _amd_partition,
    "habana": _habana_partition,
    "unknown": _cpu_only_partition,
}


def create_partition(
    accelerator: str | None,
    device_count: int,
    gpus_per_worker: int = 1,
    mig: str | tuple[str, ...] | None = None,
) -> PartitionStrategy:
    """Create partition strategy for accelerator type.

    This is the main entry point for creating device partitioning strategies.
    It dispatches to the appropriate handler based on accelerator family.

    Args:
        accelerator: GPU/accelerator type (e.g., "H100-80GB", "T4").
                    None for CPU-only workloads.
        device_count: Total devices on instance.
        gpus_per_worker: GPUs per worker for multi-GPU setups. Ignored if mig is set.
        mig: MIG profile(s) for partitioning. None = no MIG.
            - String: homogeneous (e.g., "3g.40gb" -> 2 workers per GPU)
            - Tuple: heterogeneous (e.g., ("4g.40gb", "3g.40gb") -> 2 workers)

    Returns:
        PartitionStrategy with setup script and env var function.

    Raises:
        ValueError: If configuration is invalid for accelerator type.

    Examples:
        >>> strategy = create_partition("H100-80GB", 4, gpus_per_worker=2)
        >>> strategy.get_worker_env(0, 4)
        {'CUDA_VISIBLE_DEVICES': '0,1'}

        >>> strategy = create_partition("H100-80GB", 2, mig="3g.40gb")
        >>> strategy.setup_script  # Contains nvidia-smi mig commands
        >>> strategy.workers_per_gpu
        2
    """
    if accelerator is None:
        return _cpu_only_partition("", 1, None, 0)

    family = _get_accelerator_family(accelerator)
    handler = _PARTITION_HANDLERS.get(family, _cpu_only_partition)
    return handler(accelerator, gpus_per_worker, mig, device_count)
