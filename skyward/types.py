"""Type definitions for skyward using Python 3.12+ generics."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from skyward.image import Image
    from skyward.internal.object_pool import ObjectPool
    from skyward.providers.ssh import ChannelStream, SSHConfig, SSHConnection

from skyward.accelerators import (
    GPU,
    NVIDIA,
    AcceleratorCount,
    AcceleratorSpec,
    Trainium,
    current_accelerator,
)
from skyward.exceptions import NoMatchingInstanceError

__all__ = [
    # Accelerators
    "NVIDIA",
    "Trainium",
    "GPU",
    "AcceleratorSpec",
    "current_accelerator",
    # Core types
    "InstanceSpec",
    "select_instance",
    "select_instances",
    "parse_memory_mb",
    "Instance",
    "ExitedInstance",
    # Architecture
    "Auto",
    "Architecture",
    # Memory
    "Megabytes",
    "Memory",
    # Protocols
    "ComputeSpec",
    "Provider",
    "ProviderConfig",
    # Provider selection types
    "SelectionStrategy",
    "ProviderSelector",
    "SelectionLike",
    "ProviderLike",
]

class Auto:
    pass

type Architecture = Literal['arm64', 'x86_64'] | Auto

type Megabytes = int

type Memory = Literal[
    '512MiB',
    '1GiB',
    '2GiB',
    '4GiB',
    '8GiB',
    '16GiB',
    '32GiB',
    '64GiB',
    '128GiB',
    '256GiB',
    '512GiB',
] | Megabytes | Auto


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


def parse_memory_mb(memory: Memory | None) -> int:
    """Parse memory to MB.

    Args:
        memory: Memory specification. Can be:
            - None or Auto(): Returns 1024 MB (1GB) default
            - int: Treated as MB directly
            - str: Parsed with suffix (e.g., '8GiB', '4096MB', '8G', '4096M')

    Returns:
        Memory in MB.
    """
    match memory:
        case None | Auto():
            return 1024
        case int() as mb:
            return mb
        case str() as s:
            s = s.strip().upper()
            if s.endswith("GIB"):
                return int(float(s[:-3]) * 1024)
            if s.endswith("MIB"):
                return int(float(s[:-3]))
            if s.endswith("GB"):
                return int(float(s[:-2]) * 1024)
            if s.endswith("MB"):
                return int(float(s[:-2]))
            if s.endswith("G"):
                return int(float(s[:-1]) * 1024)
            if s.endswith("M"):
                return int(float(s[:-1]))
            # Assume MB if no suffix
            return int(float(s))


def _normalize_accelerator_name(acc: str | AcceleratorSpec) -> str:
    """Normalize accelerator name for comparison."""
    acc_str = acc.accelerator if isinstance(acc, AcceleratorSpec) else str(acc)
    return acc_str.upper().replace("-", "").replace("_", "")


def _matches_count(spec_count: float, requirement: AcceleratorCount) -> bool:
    """Check if spec_count matches the requirement.

    Args:
        spec_count: Actual accelerator count from the instance spec.
        requirement: Either an exact number or a predicate function.

    Returns:
        True if spec_count matches the requirement.
    """
    if callable(requirement):
        return requirement(int(spec_count))
    return int(spec_count) == int(requirement)


def select_instance(
    instances: tuple[InstanceSpec, ...],
    cpu: int = 1,
    memory_mb: int = 1024,
    accelerator: str | AcceleratorSpec | None = None,
    accelerator_count: AcceleratorCount = 1,
    prefer_spot: bool = False,
) -> InstanceSpec:
    """Select cheapest instance that meets requirements.

    Generic selection function that works with any provider's instances.

    Args:
        instances: Available instances from provider.available_instances().
        cpu: Required number of vCPUs.
        memory_mb: Required memory in MB.
        accelerator: Required accelerator type (e.g., "H100", "A100").
        accelerator_count: Required number of accelerators. Can be:
            - int: Exact count (e.g., 2 means exactly 2 GPUs)
            - Callable[[int], bool]: Predicate function (e.g., lambda c: c >= 2)
        prefer_spot: If True, sort by spot price; otherwise by on-demand price.

    Returns:
        InstanceSpec that meets requirements (cheapest option).

    Raises:
        ValueError: If no instance type meets the requirements.
    """
    memory_gb = memory_mb / 1024

    def _price_key(s: InstanceSpec) -> float:
        if prefer_spot:
            return s.price_spot or s.price_on_demand or float("inf")
        return s.price_on_demand or float("inf")

    if accelerator:
        acc_normalized = _normalize_accelerator_name(accelerator)
        candidates = [
            s for s in instances
            if s.accelerator
            and _normalize_accelerator_name(s.accelerator).startswith(acc_normalized)
            and _matches_count(s.accelerator_count, accelerator_count)
            and s.vcpu >= cpu
            and s.memory_gb >= memory_gb
        ]

        if not candidates:
            available_counts = sorted({
                int(s.accelerator_count) for s in instances
                if s.accelerator
                and _normalize_accelerator_name(s.accelerator).startswith(acc_normalized)
            })
            if available_counts:
                raise NoMatchingInstanceError(
                    f"No instance type found for {accelerator_count}x {accelerator} "
                    f"with {cpu} vCPU, {memory_gb}GB RAM. "
                    f"Available counts for {accelerator}: {available_counts}"
                )
            available = sorted({s.accelerator for s in instances if s.accelerator})
            raise NoMatchingInstanceError(
                f"No instance type found for {accelerator}. "
                f"Available accelerator types: {available}"
            )

        candidates.sort(key=_price_key)
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
        raise NoMatchingInstanceError(
            f"No instance type found for {cpu} vCPU, {memory_gb}GB RAM. "
            f"Maximum available: {max_vcpu} vCPU, {max_mem}GB RAM"
        )

    candidates.sort(key=_price_key)
    return candidates[0]


def select_instances(
    instances: tuple[InstanceSpec, ...],
    cpu: int = 1,
    memory_mb: int = 1024,
    accelerator: str | AcceleratorSpec | None = None,
    accelerator_count: AcceleratorCount = 1,
    prefer_spot: bool = False,
) -> tuple[InstanceSpec, ...]:
    """Select all instances that meet requirements, sorted by price.

    Similar to select_instance but returns ALL matching candidates instead of
    just the cheapest. Useful for Fleet API which can try multiple instance
    types and pick one with available capacity.

    Args:
        instances: Available instances from provider.available_instances().
        cpu: Required number of vCPUs.
        memory_mb: Required memory in MB.
        accelerator: Required accelerator type (e.g., "H100", "A100").
        accelerator_count: Required number of accelerators. Can be:
            - int: Exact count (e.g., 2 means exactly 2 GPUs)
            - Callable[[int], bool]: Predicate function (e.g., lambda c: c >= 2)
        prefer_spot: If True, sort by spot price; otherwise by on-demand price.

    Returns:
        Tuple of InstanceSpecs that meet requirements, sorted by price (cheapest first).

    Raises:
        ValueError: If no instance type meets the requirements.
    """
    memory_gb = memory_mb / 1024

    def _price_key(s: InstanceSpec) -> float:
        if prefer_spot:
            return s.price_spot or s.price_on_demand or float("inf")
        return s.price_on_demand or float("inf")

    if accelerator:
        acc_normalized = _normalize_accelerator_name(accelerator)
        candidates = [
            s for s in instances
            if s.accelerator
            and _normalize_accelerator_name(s.accelerator).startswith(acc_normalized)
            and _matches_count(s.accelerator_count, accelerator_count)
            and s.vcpu >= cpu
            and s.memory_gb >= memory_gb
        ]

        if not candidates:
            available_counts = sorted({
                int(s.accelerator_count) for s in instances
                if s.accelerator
                and _normalize_accelerator_name(s.accelerator).startswith(acc_normalized)
            })
            if available_counts:
                raise NoMatchingInstanceError(
                    f"No instance type found for {accelerator_count}x {accelerator} "
                    f"with {cpu} vCPU, {memory_gb}GB RAM. "
                    f"Available counts for {accelerator}: {available_counts}"
                )
            available = sorted({s.accelerator for s in instances if s.accelerator})
            raise NoMatchingInstanceError(
                f"No instance type found for {accelerator}. "
                f"Available accelerator types: {available}"
            )

        candidates.sort(key=_price_key)
        return tuple(candidates)

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
        raise NoMatchingInstanceError(
            f"No instance type found for {cpu} vCPU, {memory_gb}GB RAM. "
            f"Maximum available: {max_vcpu} vCPU, {max_mem}GB RAM"
        )

    candidates.sort(key=_price_key)
    return tuple(candidates)


@dataclass
class Instance:
    """Represents a provisioned compute instance.

    Stores both common fields and provider-specific metadata.
    Metadata is stored as frozenset for immutability.
    """

    id: str
    provider: Provider
    ssh: SSHConfig = field(repr=False)
    spot: bool = False
    private_ip: str = ""
    public_ip: str | None = None
    node: int = 0  # 0 = head node

    # Provider-specific metadata as frozen key-value pairs
    # AWS: {"instance_id": "i-xxx", "region": "us-east-1"}
    # DigitalOcean: {"droplet_id": 12345}
    metadata: frozenset[tuple[str, Any]] = field(default_factory=frozenset)

    # Lock for thread-safe pool initialization (per-instance)
    _pool_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
        compare=False,
    )

    # Metrics streamer (managed by start_metrics/stop_metrics)
    _metrics: Any = field(default=None, init=False, repr=False, compare=False)

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

    @property
    def pool(self) -> ObjectPool[SSHConnection]:
        """Thread-safe lazy SSH connection pool."""
        # Fast path: already initialized
        if "_pool" in self.__dict__:
            return self.__dict__["_pool"]

        # Slow path: initialize with lock
        with self._pool_lock:
            # Double-check after acquiring lock
            if "_pool" in self.__dict__:
                return self.__dict__["_pool"]

            from skyward.providers.ssh import SSHPool

            self.__dict__["_pool"] = SSHPool(self.ssh)
            return self.__dict__["_pool"]

    @contextmanager
    def connect(self) -> Iterator[SSHConnection]:
        """Connect with auto-commit on exit."""
        conn = self.pool.acquire()
        try:
            yield conn
            conn.commit()
        finally:
            self.pool.release(conn)

    def run_command(self, command: str, timeout: int = 30) -> str:
        """Execute single command immediately."""
        from loguru import logger

        logger.debug(f"Instance.run_command: acquiring SSH connection for {self.id}")
        with self.pool() as conn:
            logger.debug(f"Instance.run_command: got connection for {self.id}, executing")
            result = conn.exec(command, timeout)
            logger.debug(f"Instance.run_command: command completed for {self.id}")
            return result

    def start_metrics(
        self,
        interval: float = 0.2,
        provider_name: Any = None,
    ) -> None:
        """Start streaming metrics from this instance.

        Args:
            interval: Time between samples in seconds.
            provider_name: ProviderName enum for event metadata.
        """
        if self._metrics is not None:
            return  # Already streaming

        from skyward.events import ProviderName
        from skyward.metrics import MetricsStreamer

        self._metrics = MetricsStreamer(
            instance=self,
            interval=interval,
            provider_name=provider_name or ProviderName.AWS,
        )
        self._metrics.start()

    def stop_metrics(self) -> None:
        """Stop streaming metrics."""
        if self._metrics is not None:
            self._metrics.stop()
            self._metrics = None

    def close(self) -> None:
        """Close instance resources (metrics + SSH pool)."""
        self.stop_metrics()
        if "_pool" in self.__dict__:
            self.__dict__["_pool"].close_all()
            del self.__dict__["_pool"]

    @contextmanager
    def open_channel(self, remote_port: int = 18861) -> Iterator[ChannelStream]:
        """Open SSH tunnel channel to remote port.

        Uses Paramiko direct-tcpip channel - no subprocess needed.
        The channel implements RPyC Stream interface for use with rpyc.connect_stream().
        """
        conn = self.pool.acquire()
        try:
            yield conn.open_tunnel(remote_port)
        finally:
            self.pool.release(conn)

    def run_python(self, script: str, timeout: int = 30) -> str:
        return self.run_command(f"/opt/skyward/.venv/bin/python -c '{script}'", timeout)

    def metrics(self) -> dict[str, Any]:
        """Fetch metrics from instance.

        .. deprecated::
            Use MetricsStreamer for efficient streaming via RPyC.
            This method uses SSH + cloudpickle which is slow (~300-500ms per call).
            The new MetricsStreamer uses RPyC generators (~2-5ms per sample).
        """
        import warnings

        warnings.warn(
            "Instance.metrics() is deprecated. Use MetricsStreamer for efficient "
            "metrics streaming via RPyC (~60-250x faster).",
            DeprecationWarning,
            stacklevel=2,
        )

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
                query = "utilization.gpu,memory.used,memory.total,temperature.gpu"
                r = subprocess.run(
                    ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5
                )
                if r.returncode == 0:
                    parts = r.stdout.strip().split(", ")
                    if len(parts) >= 4:
                        m["gpu_utilization"] = float(parts[0])
                        m["gpu_memory_used_mb"] = float(parts[1])
                        m["gpu_memory_total_mb"] = float(parts[2])
                        m["gpu_temperature"] = float(parts[3])
            except Exception:
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
            stdout = self.run_command(f"sudo /opt/skyward/.venv/bin/python -c \"{script}\"")
            return cloudpickle.loads(base64.b64decode(stdout.strip()))  # type: ignore[return-value]

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
# Provider Selection Types
# =============================================================================

type ProviderLiteral = Literal['aws', 'verda', 'digital_ocean', 'vastai']

type SelectionStrategy = Literal["first", "cheapest", "available"]
"""Built-in provider selection strategies."""

type ProviderSelector = Callable[[tuple[Provider, ...], ComputeSpec], Provider]
"""Callable that selects a provider from a list based on compute requirements."""

type SelectionLike = SelectionStrategy | ProviderSelector
"""Either a built-in strategy name or a custom selector function."""

type SingleProvider = ProviderConfig | ProviderLiteral

type ProviderLike = SingleProvider | Sequence[SingleProvider]
"""Single provider config or sequence of configs for multi-provider selection."""
