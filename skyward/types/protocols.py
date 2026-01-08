"""Protocol definitions for skyward types."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from skyward.types.core import Architecture, Memory

if TYPE_CHECKING:
    from skyward.spec.image import Image
    from skyward.types.instance import Instance
    from skyward.types.spec import InstanceSpec, InstanceStatus

__all__ = [
    "Instances",
    "ProviderConfig",
    "ComputeSpec",
    "Provider",
]


class Instances(Protocol):
    """Protocol for accessing instances in a cluster.

    Provides iteration over current instances.
    Implementors (like InstancePool) manage lifecycle separately.
    """

    def __iter__(self) -> Iterator[Instance]: ...
    def __len__(self) -> int: ...


@runtime_checkable
class ProviderConfig(Protocol):
    """Protocol for provider configuration.

    ProviderConfig represents immutable cloud provider configuration.
    It provides a build() method to create a stateful Provider instance.

    Example:
        config = AWS(region="us-east-1")  # Pure config, no side effects
        provider = config.build()          # Creates stateful provider

        # Or let ComputePool build it:
        pool = ComputePool(provider=config)  # build() called in __enter__
    """

    def build(self) -> Provider:
        """Build a stateful Provider from this configuration."""
        ...


@runtime_checkable
class ComputeSpec(Protocol):
    """Protocol for compute specifications passed to providers.

    This defines the interface that providers expect for provisioning,
    setup, and shutdown phases. Implemented by _PoolCompute.
    """

    nodes: int
    machine: str | None  # Direct instance type override (e.g., "p5.48xlarge")
    accelerator: Any  # Accelerator | list[Accelerator] | str (e.g., "H100:8:16")
    architecture: Architecture  # Auto (cheapest) or explicit arm64/x86_64
    image: Image  # Environment specification (python, pip, apt, env)
    cpu: int | None
    memory: Memory | None
    timeout: int
    allocation: Any  # AllocationLike
    volumes: list[Any]  # list[Volume]
    max_hourly_cost: float | None  # USD per hour for entire cluster
    concurrency: int  # Concurrent tasks per node (for SSH pool sizing)

    @property
    def name(self) -> str:
        """Compute name for identification."""
        ...

    @property
    def is_cluster(self) -> bool:
        """True if multi-node cluster."""
        ...

    @property
    def wrapped_fn(self) -> Callable[..., Any]:
        """The wrapped function to execute."""
        ...

    @property
    def head_port(self) -> int:
        """Port for distributed head node communication."""
        ...

    @property
    def placement_group(self) -> str | None:
        """Placement group name for co-located instances."""
        ...

    @property
    def workers_per_instance(self) -> int:
        """Number of workers per instance (for worker isolation)."""
        ...


@runtime_checkable
class Provider(Protocol):
    """Protocol for cloud providers.

    Providers create and manage cloud compute instances:
    - provision: Create instances with _destroy_fn callbacks
    - discover_peers: Find instances in a cluster
    - available_instances: List available instance types
    - cleanup: Clean up provider-level resources (optional)

    Instance lifecycle (bootstrap, shutdown) is managed by Instance methods.
    Events are emitted via the callback system (see skyward.callback).
    """

    @property
    def name(self) -> str:
        """Provider name (aws, digitalocean, etc.)."""
        ...

    def provision(
        self,
        compute: ComputeSpec,
    ) -> tuple[Instance, ...]:
        """Provision compute instances.

        Creates instances and injects _destroy_fn callback for each.
        Emits ProvisioningStarted/ProvisioningCompleted events.

        Args:
            compute: Compute specification including nodes, accelerator, etc.

        Returns:
            Tuple of provisioned Instance objects with _destroy_fn set.
        """
        ...

    def discover_peers(self, cluster_id: str) -> tuple[Instance, ...]:
        """Discover peer instances in a cluster.

        Queries the cloud API for all running instances with the given cluster_id.
        Used for autodiscovery in distributed actor systems.

        Args:
            cluster_id: Unique identifier for the cluster (job_id).

        Returns:
            Tuple of Instance objects for all running peers, sorted by private_ip.
            The node field is assigned based on sort order (lowest IP = node 0).
        """
        ...

    def available_instances(self) -> tuple[InstanceSpec, ...]:
        """List all available instance types for this provider.

        Returns cached results from discovery (24h TTL).
        Includes both standard (CPU) and accelerator (GPU) instances.

        Returns:
            Tuple of InstanceSpec with available instance types.
        """
        ...

    def cleanup(self) -> None:
        """Clean up provider-level resources after all instances are destroyed.

        Called after all instances have been destroyed via Instance.destroy().
        Used for cleaning up cluster-level resources like overlay networks.

        Default implementation does nothing. Override if provider needs cleanup.
        """
        ...

    def get_instance_status(self, instance_id: str) -> InstanceStatus | None:
        """Get current status of an instance.

        Used by preemption monitoring to detect when instances are interrupted.

        Args:
            instance_id: Instance identifier.

        Returns:
            InstanceStatus with current state, or None if instance not found.
        """
        ...

    def classify_preemption(self, status: str) -> str | None:
        """Classify if a status string indicates preemption.

        Maps provider-specific status values to preemption reasons.

        Args:
            status: Provider-specific status string (e.g., "outbid", "exited").

        Returns:
            Preemption reason ("outbid", "capacity", "maintenance") or None
            if the status does not indicate preemption.
        """
        ...
