"""EC2 instance type selection based on resource requirements."""

from __future__ import annotations

from dataclasses import dataclass

from skyward.accelerator import Accelerator


@dataclass(frozen=True)
class InstanceSpec:
    """Specification for an EC2 instance type."""

    type_name: str
    vcpu: int
    memory_gb: float
    accelerator: Accelerator = None
    accelerator_count: int = 0  # GPUs for NVIDIA, NeuronCores for Trainium
    supports_cluster_placement: bool = False
    accelerator_memory_gb: float = 0  # Memory per accelerator device


# Standard compute instances (sorted by resources ascending)
STANDARD_INSTANCES: list[InstanceSpec] = [
    InstanceSpec("t3.micro", 2, 1),
    InstanceSpec("t3.small", 2, 2),
    InstanceSpec("t3.medium", 2, 4),
    InstanceSpec("t3.large", 2, 8),
    InstanceSpec("t3.xlarge", 4, 16),
    InstanceSpec("t3.2xlarge", 8, 32),
    InstanceSpec("m5.large", 2, 8),
    InstanceSpec("m5.xlarge", 4, 16),
    InstanceSpec("m5.2xlarge", 8, 32),
    InstanceSpec("m5.4xlarge", 16, 64),
    InstanceSpec("m5.8xlarge", 32, 128),
    InstanceSpec("m5.12xlarge", 48, 192),
    InstanceSpec("m5.16xlarge", 64, 256),
    InstanceSpec("m5.24xlarge", 96, 384),
]

# Accelerator instances (GPUs and Trainium)
# All accelerator instances support cluster placement groups
ACCELERATOR_INSTANCES: list[InstanceSpec] = [
    # NVIDIA T4 instances (g4dn family) - 16GB per GPU
    InstanceSpec("g4dn.xlarge", 4, 16, "T4", 1, True, 16),
    InstanceSpec("g4dn.2xlarge", 8, 32, "T4", 1, True, 16),
    InstanceSpec("g4dn.4xlarge", 16, 64, "T4", 1, True, 16),
    InstanceSpec("g4dn.8xlarge", 32, 128, "T4", 1, True, 16),
    InstanceSpec("g4dn.12xlarge", 48, 192, "T4", 4, True, 16),
    InstanceSpec("g4dn.16xlarge", 64, 256, "T4", 1, True, 16),
    # NVIDIA L4 instances (g6 family) - 24GB per GPU
    InstanceSpec("g6.xlarge", 4, 16, "L4", 1, True, 24),
    InstanceSpec("g6.2xlarge", 8, 32, "L4", 1, True, 24),
    InstanceSpec("g6.4xlarge", 16, 64, "L4", 1, True, 24),
    InstanceSpec("g6.8xlarge", 32, 128, "L4", 1, True, 24),
    InstanceSpec("g6.12xlarge", 48, 192, "L4", 4, True, 24),
    InstanceSpec("g6.16xlarge", 64, 256, "L4", 1, True, 24),
    InstanceSpec("g6.24xlarge", 96, 384, "L4", 4, True, 24),
    InstanceSpec("g6.48xlarge", 192, 768, "L4", 8, True, 24),
    # NVIDIA A10G instances (g5 family) - 24GB per GPU
    InstanceSpec("g5.xlarge", 4, 16, "A10g", 1, True, 24),
    InstanceSpec("g5.2xlarge", 8, 32, "A10g", 1, True, 24),
    InstanceSpec("g5.4xlarge", 16, 64, "A10g", 1, True, 24),
    InstanceSpec("g5.8xlarge", 32, 128, "A10g", 1, True, 24),
    InstanceSpec("g5.12xlarge", 48, 192, "A10g", 4, True, 24),
    InstanceSpec("g5.16xlarge", 64, 256, "A10g", 1, True, 24),
    InstanceSpec("g5.24xlarge", 96, 384, "A10g", 4, True, 24),
    InstanceSpec("g5.48xlarge", 192, 768, "A10g", 8, True, 24),
    # NVIDIA L40S instances (g6e family) - 48GB per GPU
    InstanceSpec("g6e.xlarge", 4, 64, "L40S", 1, True, 48),
    InstanceSpec("g6e.2xlarge", 8, 64, "L40S", 1, True, 48),
    InstanceSpec("g6e.4xlarge", 16, 128, "L40S", 1, True, 48),
    InstanceSpec("g6e.8xlarge", 32, 256, "L40S", 1, True, 48),
    InstanceSpec("g6e.12xlarge", 48, 384, "L40S", 4, True, 48),
    InstanceSpec("g6e.16xlarge", 48, 768, "L40S", 2, True, 48),
    InstanceSpec("g6e.24xlarge", 96, 768, "L40S", 4, True, 48),
    InstanceSpec("g6e.48xlarge", 192, 1536, "L40S", 8, True, 48),
    # NVIDIA A100 instances (p4d family) - 80GB per GPU
    InstanceSpec("p4d.24xlarge", 96, 1152, "A100-80", 8, True, 80),
    # NVIDIA H100 instances (p5 family) - 80GB per GPU
    InstanceSpec("p5.48xlarge", 192, 2048, "H100", 8, True, 80),
    # AWS Trainium1 instances (trn1 family) - 16GB per NeuronCore (32GB per chip)
    # accelerator_count = NeuronCores (2 per chip)
    InstanceSpec("trn1.2xlarge", 8, 32, "Trainium1", 2, True, 16),
    InstanceSpec("trn1.32xlarge", 128, 512, "Trainium1", 32, True, 16),
    InstanceSpec("trn1n.32xlarge", 128, 512, "Trainium1", 32, True, 16),  # 2x network bandwidth
    # AWS Trainium2 instances (trn2 family) - 48GB per NeuronCore (96GB per chip)
    InstanceSpec("trn2.48xlarge", 192, 2048, "Trainium2", 32, True, 48),
]


def select_instance_type(
    cpu: int,
    memory_mb: int,
    accelerator: Accelerator = None,
) -> str:
    """Select the smallest instance type that meets requirements.

    Args:
        cpu: Required number of vCPUs.
        memory_mb: Required memory in MB.
        accelerator: Required accelerator type (e.g., "T4", "A100-80", "Trainium1").

    Returns:
        EC2 instance type name.

    Raises:
        ValueError: If no instance type meets the requirements.
    """
    memory_gb = memory_mb / 1024

    if accelerator:
        candidates = [
            inst
            for inst in ACCELERATOR_INSTANCES
            if inst.accelerator == accelerator and inst.vcpu >= cpu and inst.memory_gb >= memory_gb
        ]
        if not candidates:
            available = sorted({inst.accelerator for inst in ACCELERATOR_INSTANCES if inst.accelerator})
            raise ValueError(
                f"No instance type found for {accelerator} with {cpu} vCPU, {memory_gb}GB RAM. "
                f"Available accelerator types: {available}"
            )
        return candidates[0].type_name

    # For standard instances, find the smallest that meets requirements
    candidates = [
        inst
        for inst in STANDARD_INSTANCES
        if inst.vcpu >= cpu and inst.memory_gb >= memory_gb
    ]
    if not candidates:
        max_vcpu = max(inst.vcpu for inst in STANDARD_INSTANCES)
        max_mem = max(inst.memory_gb for inst in STANDARD_INSTANCES)
        raise ValueError(
            f"No instance type found for {cpu} vCPU, {memory_gb}GB RAM. "
            f"Maximum available: {max_vcpu} vCPU, {max_mem}GB RAM"
        )

    # Return the first matching instance (list is sorted by size)
    return candidates[0].type_name


def get_instance_spec(instance_type: str) -> InstanceSpec | None:
    """Get the specification for a specific instance type.

    Args:
        instance_type: EC2 instance type name (e.g., "t3.large", "g4dn.xlarge").

    Returns:
        InstanceSpec if found, None otherwise.
    """
    for inst in STANDARD_INSTANCES:
        if inst.type_name == instance_type:
            return inst

    for inst in ACCELERATOR_INSTANCES:
        if inst.type_name == instance_type:
            return inst

    return None


def get_instance_types_for_accelerators(accelerators: list[str]) -> list[str]:
    """Map multiple accelerators to instance types for EC2 Fleet.

    Args:
        accelerators: List of accelerator names (e.g., ["L4", "A10G", "T4"]).

    Returns:
        List of instance type names in the same order.

    Raises:
        ValueError: If any accelerator is not found.
    """
    instance_types = []
    for acc in accelerators:
        # Normalize case (e.g., "a10g" -> "A10g")
        acc_normalized = acc.upper() if acc.upper() in ("T4", "L4", "L40", "L40S") else acc
        if acc.lower() == "a10g":
            acc_normalized = "A10g"

        matching = [i for i in ACCELERATOR_INSTANCES if i.accelerator == acc_normalized]
        if not matching:
            available = sorted({i.accelerator for i in ACCELERATOR_INSTANCES if i.accelerator})
            raise ValueError(
                f"Unknown accelerator: {acc!r}. Available: {available}"
            )
        # Use smallest instance type for this accelerator
        instance_types.append(matching[0].type_name)

    return instance_types
