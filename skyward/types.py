"""Type definitions for skyward using Python 3.12+ generics."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

if TYPE_CHECKING:
    from skyward.skyimage import Image

# Re-export from accelerator module for convenience
from skyward.accelerator import GPU, NVIDIA, Accelerator, Trainium, current_accelerator

__all__ = [
    # Accelerators
    "NVIDIA",
    "Trainium",
    "GPU",
    "Accelerator",
    "current_accelerator",
    # Core types
    "InstanceSpec",
    "select_instance",
    "Instance",
    "ExitedInstance",
    # Legacy types (used by runtime modules)
    "NodeInfo",
    "ClusterTopology",
    # Protocols
    "ComputeSpec",
    "Provider",
]


@dataclass(frozen=True, slots=True)
class InstanceSpec:
    """Unified instance specification across all providers.

    Represents an available instance/droplet type with its specs.
    Provider-specific data (e.g., architecture, regions) goes in metadata.
    """

    name: str  # type_name (AWS) ou slug (DO, Verda)
    vcpu: int
    memory_gb: float
    accelerator: str | None = None
    accelerator_count: float = 0
    accelerator_memory_gb: float = 0
    price_on_demand: float | None = None  # USD per hour
    price_spot: float | None = None  # USD per hour (if available)
    billing_increment_minutes: int | None = None  # None = per-second billing
    metadata: dict[str, Any] = field(default_factory=dict)


def _normalize_accelerator_name(acc: str | Accelerator) -> str:
    """Normalize accelerator name for comparison."""
    acc_str = acc.accelerator if isinstance(acc, Accelerator) else str(acc)
    return acc_str.upper().replace("-", "").replace("_", "")


def select_instance(
    instances: tuple[InstanceSpec, ...],
    cpu: int = 1,
    memory_mb: int = 1024,
    accelerator: str | Accelerator | None = None,
    accelerator_count: float = 1,
) -> InstanceSpec:
    """Select smallest instance that meets requirements.

    Generic selection function that works with any provider's instances.

    Args:
        instances: Available instances from provider.available_instances().
        cpu: Required number of vCPUs.
        memory_mb: Required memory in MB.
        accelerator: Required accelerator type (e.g., "H100", "A100").
        accelerator_count: Required number of accelerators.

    Returns:
        InstanceSpec that meets requirements.

    Raises:
        ValueError: If no instance type meets the requirements.
    """
    memory_gb = memory_mb / 1024

    if accelerator:
        acc_normalized = _normalize_accelerator_name(accelerator)
        candidates = [
            s for s in instances
            if s.accelerator
            and _normalize_accelerator_name(s.accelerator) == acc_normalized
            and s.accelerator_count >= accelerator_count
            and s.vcpu >= cpu
            and s.memory_gb >= memory_gb
        ]

        if not candidates:
            available_counts = sorted({
                s.accelerator_count for s in instances
                if s.accelerator
                and _normalize_accelerator_name(s.accelerator) == acc_normalized
            })
            if available_counts:
                raise ValueError(
                    f"No instance type found for {accelerator_count}x {accelerator} "
                    f"with {cpu} vCPU, {memory_gb}GB RAM. "
                    f"Available counts for {accelerator}: {available_counts}"
                )
            available = sorted({s.accelerator for s in instances if s.accelerator})
            raise ValueError(
                f"No instance type found for {accelerator}. "
                f"Available accelerator types: {available}"
            )

        return candidates[0]

    # Standard instances (no accelerator)
    candidates = [
        s for s in instances
        if not s.accelerator
        and s.vcpu >= cpu
        and s.memory_gb >= memory_gb
    ]

    if not candidates:
        # Fallback: include any instance that meets CPU/memory
        candidates = [
            s for s in instances
            if s.vcpu >= cpu and s.memory_gb >= memory_gb
        ]

    if not candidates:
        max_vcpu = max((s.vcpu for s in instances), default=0)
        max_mem = max((s.memory_gb for s in instances), default=0)
        raise ValueError(
            f"No instance type found for {cpu} vCPU, {memory_gb}GB RAM. "
            f"Maximum available: {max_vcpu} vCPU, {max_mem}GB RAM"
        )

    return candidates[0]


@dataclass(frozen=True, slots=True)
class Instance:
    """Represents a provisioned compute instance.

    Stores both common fields and provider-specific metadata.
    Metadata is stored as frozenset for immutability.
    """

    id: str
    provider: Provider
    spot: bool = False
    private_ip: str = ""
    public_ip: str | None = None
    node: int = 0  # 0 = head node

    # Provider-specific metadata as frozen key-value pairs
    # AWS: {"instance_id": "i-xxx", "region": "us-east-1"}
    # DigitalOcean: {"droplet_id": 12345}
    metadata: frozenset[tuple[str, Any]] = field(default_factory=frozenset)

    @property
    def is_head(self) -> bool:
        """True if this is the head node."""
        return self.node == 0

    def get_meta(self, key: str, default: Any = None) -> Any:
        """Get provider-specific metadata by key."""
        for k, v in self.metadata:
            if k == key:
                return v
        return default

    def run_command(self, command: str, timeout: int = 30) -> str:
        return self.provider.run_command(self, command, timeout)

    def run_python(self, script: str, timeout: int = 30) -> str:
        return self.run_command(f"/opt/skyward/.venv/bin/python -c '{script}'", timeout)

    def metrics(self) -> dict[str, Any]:
        @self.remote
        def fetch_metrics() -> dict[str, Any]:
            import subprocess

            import psutil

            mem = psutil.virtual_memory()
            m = {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": mem.percent,
                "memory_used_mb": mem.used / (1024 * 1024),
                "memory_total_mb": mem.total / (1024 * 1024),
            }

            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5
                )
                if r.returncode == 0:
                    parts = r.stdout.strip().split(", ")
                    if len(parts) >= 4:
                        m["gpu_utilization"] = float(parts[0])
                        m["gpu_memory_used_mb"] = float(parts[1])
                        m["gpu_memory_total_mb"] = float(parts[2])
                        m["gpu_temperature"] = float(parts[3])
            except:
                pass

            return m

        return fetch_metrics()


    def remote[**P, R](self, fn: Callable[P, R]) -> Callable[P, R]:
        """Execute function remotely via shell + cloudpickle.

        Args:
            fn: Function to execute remotely.

        Returns:
            Callable that executes fn on this instance.

        Example:
            def add(a: int, b: int) -> int:
                return a + b

            result = instance.remote(add)(1, 2)  # Returns 3
        """
        import base64
        from functools import wraps

        import cloudpickle

        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            fn_b64 = base64.b64encode(cloudpickle.dumps(fn)).decode()
            args_b64 = base64.b64encode(cloudpickle.dumps(args)).decode()
            kwargs_b64 = base64.b64encode(cloudpickle.dumps(kwargs)).decode()

            script = (
                "import sys,base64,cloudpickle;"
                f"fn=cloudpickle.loads(base64.b64decode('{fn_b64}'));"
                f"args=cloudpickle.loads(base64.b64decode('{args_b64}'));"
                f"kwargs=cloudpickle.loads(base64.b64decode('{kwargs_b64}'));"
                "r=fn(*args,**kwargs);"
                "print(base64.b64encode(cloudpickle.dumps(r)).decode())"
            )
            stdout = self.run_command(f"/opt/skyward/.venv/bin/python -c \"{script}\"")
            return cast(R, cloudpickle.loads(base64.b64decode(stdout.strip())))

        return wrapper

@dataclass(frozen=True, slots=True)
class ExitedInstance:
    """Represents an instance that has been shut down."""

    instance: Instance
    exit_code: int | None = None
    exit_reason: str = ""  # "normal", "spot_interruption", "timeout", "error"
    error_message: str | None = None
    duration_seconds: float = 0.0


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class ComputeSpec(Protocol):
    """Protocol for compute specifications passed to providers.

    This defines the interface that providers expect for provisioning,
    setup, and shutdown phases. Implemented by _PoolCompute.
    """

    nodes: int
    accelerator: Any  # Accelerator | list[Accelerator] | str (e.g., "H100:8:16")
    image: Image  # Environment specification (python, pip, apt, env)
    cpu: int | None
    memory: str | None
    timeout: int
    spot: Any  # SpotLike
    volumes: list[Any]  # list[Volume]

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
    """Protocol for cloud providers with phase-based lifecycle.

    Providers implement three distinct phases:
    - provision: Create compute instances
    - setup: Bootstrap instances with dependencies
    - shutdown: Clean up resources

    Execution is handled directly by ComputePool via RPyC connections.

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

        Args:
            compute: Compute specification including nodes, accelerator, etc.

        Returns:
            Tuple of provisioned Instance objects.
        """
        ...

    def setup(
        self,
        instances: tuple[Instance, ...],
        compute: ComputeSpec,
    ) -> None:
        """Setup instances with dependencies and skyward.

        Args:
            instances: Instances from provision phase.
            compute: Compute specification.
        """
        ...

    def shutdown(
        self,
        instances: tuple[Instance, ...],
        compute: ComputeSpec,
    ) -> tuple[ExitedInstance, ...]:
        """Shutdown instances.

        Args:
            instances: Instances to shut down.
            compute: Compute specification.

        Returns:
            Tuple of ExitedInstance objects.
        """
        ...

    def create_tunnel(
        self,
        instance: Instance,
        remote_port: int = 18861,
    ) -> tuple[int, subprocess.Popen[bytes]]:
        """Create tunnel to instance for RPyC connection.

        Args:
            instance: Instance to connect to.
            remote_port: Remote port to tunnel to (default: 18861).
                        For worker isolation, use 18861 + worker_id.

        Returns:
            Tuple of (local_port, tunnel_process).
        """
        ...

    def run_command(self, instance: Instance, command: str, timeout: int = 30) -> str:
        """Run shell command on instance, return stdout.

        This method abstracts the transport layer (SSM for AWS, SSH for
        DigitalOcean) allowing common scripts to be executed on any provider.

        Args:
            instance: Target instance.
            command: Shell command string to execute.
            timeout: Maximum time to wait in seconds.

        Returns:
            stdout from the command.

        Raises:
            RuntimeError: If command fails (non-zero exit code).
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


# =============================================================================
# Legacy Types (for compatibility during migration)
# =============================================================================


@dataclass(frozen=True, slots=True)
class NodeInfo:
    """Information about a provisioned node."""

    id: str
    private_ip: str
    public_ip: str | None = None
    node: int = 0

    @property
    def is_head(self) -> bool:
        """True if this is the head node."""
        return self.node == 0


@dataclass(frozen=True, slots=True)
class ClusterTopology:
    """Complete cluster topology."""

    cluster_id: str
    nodes: tuple[NodeInfo, ...]
    head_addr: str
    head_port: int

    @property
    def total_nodes(self) -> int:
        """Total number of nodes in the cluster."""
        return len(self.nodes)

    @property
    def head(self) -> NodeInfo:
        """Returns the head node."""
        return self.nodes[0]

    def get_node(self, node: int) -> NodeInfo:
        """Returns the node with the specified index."""
        return self.nodes[node]
