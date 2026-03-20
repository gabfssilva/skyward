"""Specification dataclasses for pool and image configuration.

These are the immutable configuration objects that define what
the user wants. Components use these specs to provision resources.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypedDict

from skyward.api.metrics import Metric, MetricsConfig

if TYPE_CHECKING:
    from skyward.accelerators import Accelerator
    from skyward.actors.messages import ProviderName
    from skyward.api.logging import LogConfig
    from skyward.api.plugin import Plugin
    from skyward.api.provider import ProviderConfig
    from skyward.storage import Storage


type AllocationStrategy = Literal[
    "spot",
    "on-demand",
    "spot-if-available",
    "cheapest",
]
"""Instance lifecycle strategy.

- ``"spot"`` -- spot/preemptible only (cheapest, may be interrupted).
- ``"on-demand"`` -- on-demand only (reliable, higher cost).
- ``"spot-if-available"`` -- try spot first, fall back to on-demand.
- ``"cheapest"`` -- compare spot and on-demand, pick lowest price.
"""

type Architecture = Literal["x86_64", "arm64"]
"""CPU architecture filter."""

type SelectionStrategy = Literal["first", "cheapest"]
"""Multi-spec selection strategy.

- ``"first"`` -- use the first spec that has available offers.
- ``"cheapest"`` -- compare all specs, pick the lowest price.
"""

type SkywardSource = Literal["auto", "local", "github", "pypi"]
"""Skyward installation source for remote workers."""

type WorkerExecutor = Literal["auto", "thread", "process"]
"""Worker execution backend.

- ``"auto"`` -- resolves to ``"thread"`` (default).
- ``"thread"`` -- ThreadPoolExecutor. Supports streaming, distributed
  collections without IPC, and has lower overhead.
- ``"process"`` -- ProcessPoolExecutor. Useful for CPU-bound pure-Python
  workloads that benefit from bypassing the GIL.
"""


@dataclass(frozen=True, slots=True)
class Nodes:
    """Node count specification.

    Parameters
    ----------
    min
        Number of nodes to provision. Must be >= 1.
    max
        Maximum node count for autoscaling. ``None`` disables autoscaling.
    desired
        Minimum nodes needed before pool becomes operational.
        ``None`` waits for all ``min`` nodes.
    """

    min: int
    max: int | None = None
    desired: int | None = None

    @property
    def auto_scaling(self) -> bool:
        """Whether autoscaling is enabled."""
        return self.max is not None

    def __post_init__(self) -> None:
        if self.min < 1:
            raise ValueError(f"min must be >= 1, got {self.min}")
        if self.max is not None and self.max < self.min:
            raise ValueError(f"max ({self.max}) must be >= min ({self.min})")
        upper = self.max if self.max is not None else self.min
        effective_desired = self.desired if self.desired is not None else self.min
        if effective_desired > upper:
            raise ValueError(
                f"desired ({effective_desired}) must be <= {'max' if self.max is not None else 'min'} ({upper})"
            )
        if effective_desired < 1:
            raise ValueError(f"desired must be >= 1, got {effective_desired}")


type NodeSpec = int | tuple[int, int] | Nodes


@dataclass(frozen=True, slots=True)
class Worker:
    """Worker configuration per node.

    Parameters
    ----------
    concurrency
        Number of concurrent task slots per node. Default ``1``.
    executor
        Execution backend -- ``"auto"`` (default) resolves to ``"thread"``.
        Use ``"process"`` explicitly for CPU-bound pure-Python workloads
        that benefit from bypassing the GIL. Thread executor supports
        streaming, distributed collections without IPC, and has lower overhead.
    """

    concurrency: int = 1
    executor: WorkerExecutor = "auto"

    @property
    def resolved_executor(self) -> Literal["thread", "process"]:
        """Resolve "auto" to a concrete executor."""
        match self.executor:
            case "auto":
                return "thread"
            case concrete:
                return concrete


@dataclass(frozen=True, slots=True)
class Spec:
    """Hardware and environment specification for compute pools.

    Pass one or more ``Spec`` objects to ``sky.Compute`` to define
    what to provision. For multi-provider fallback, pass multiple specs
    and set ``selection`` in ``Options``.

    Parameters
    ----------
    provider
        Cloud provider configuration (e.g., ``sky.AWS()``, ``sky.VastAI()``).
    accelerator
        GPU type (e.g., ``"A100"``, ``"H100"``). ``None`` for CPU-only.
    nodes
        Fixed node count or ``(min, max)`` tuple for autoscaling.
    vcpus
        Minimum vCPUs per node.
    memory_gb
        Minimum RAM in GB per node.
    architecture
        CPU architecture filter. ``None`` accepts any.
    allocation
        Instance lifecycle strategy.
    image
        Environment specification (Python version, packages, etc.).
    region
        Cloud region (provider-specific).
    max_hourly_cost
        Cost cap per node per hour in USD.
    ttl
        Auto-shutdown timeout in seconds after pool exits. ``0`` disables.
    volumes
        S3/GCS volumes to mount on workers.
    plugins
        Composable plugins to apply to the pool.

    Examples
    --------
    >>> with sky.Compute(
    ...     sky.Spec(provider=sky.VastAI(), accelerator="A100"),
    ...     sky.Spec(provider=sky.AWS(), accelerator="A100"),
    ... ) as compute:
    ...     result = train(data) >> compute
    """
    provider: ProviderConfig
    accelerator: Accelerator | None = None
    nodes: NodeSpec = 1
    vcpus: float | None = None
    memory_gb: float | None = None
    disk_gb: int | None = None
    architecture: Architecture | None = None
    allocation: AllocationStrategy = "spot-if-available"
    image: Image = field(default_factory=lambda: Image())
    region: str | None = None
    max_hourly_cost: float | None = None
    ttl: int = 600
    volumes: list[Volume] | tuple[Volume, ...] = ()
    plugins: list[Plugin] | tuple[Plugin, ...] = ()


class _SpecRequired(TypedDict):
    provider: ProviderConfig


class SpecKwargs(_SpecRequired, total=False):
    """TypedDict mirror of ``Spec`` for ``Unpack`` in function signatures.

    Allows ``sky.Compute(provider=sky.AWS(), accelerator="A100")``
    with full type safety. Convert to ``Spec`` via ``Spec(**kwargs)``.
    """

    accelerator: Accelerator | None
    nodes: NodeSpec
    vcpus: float | None
    memory_gb: float | None
    disk_gb: int | None
    architecture: Architecture | None
    allocation: AllocationStrategy
    image: Image
    region: str | None
    max_hourly_cost: float | None
    ttl: int
    volumes: list[Volume] | tuple[Volume, ...]
    plugins: list[Plugin] | tuple[Plugin, ...]


@dataclass(frozen=True, slots=True)
class Options:
    """Operational tuning for compute pools.

    Groups all operational parameters (timeouts, retries, autoscaling,
    session settings) into a single object. Sensible defaults mean most
    users never need to create one explicitly.

    Parameters
    ----------
    selection
        Multi-spec selection strategy.
    worker
        Worker configuration (concurrency, executor).
    provision_timeout
        Maximum seconds to wait for provisioning.
    provision_retry_delay
        Seconds between provision retry attempts.
    max_provision_attempts
        Maximum number of provision attempts.
    ssh_timeout
        SSH connection timeout in seconds.
    ssh_retry_interval
        Seconds between SSH retry attempts.
    default_compute_timeout
        Default timeout in seconds for submitted tasks.
    autoscale_cooldown
        Seconds between autoscaling decisions.
    autoscale_idle_timeout
        Seconds of idle before autoscaler scales down.
    reconcile_tick_interval
        Seconds between reconciler ticks.
    shutdown_timeout
        Maximum seconds to wait for a graceful shutdown.
    console
        Enable the Rich adaptive console spy.
    logging
        Logging configuration. ``True`` uses sensible defaults,
        ``False`` disables logging, or pass a ``LogConfig`` instance.

    Examples
    --------
    >>> with sky.Compute(
    ...     sky.Spec(provider=sky.AWS(), accelerator="A100", nodes=4),
    ...     options=sky.Options(provision_timeout=600, console=False),
    ... ) as compute:
    ...     result = train(data) >> compute
    """

    selection: SelectionStrategy = "cheapest"
    worker: Worker | None = None
    provision_timeout: int = 600
    provision_retry_delay: float = 5.0
    max_provision_attempts: int = 3
    ssh_timeout: int = 300
    ssh_retry_interval: int = 2
    default_compute_timeout: float = 300.0
    autoscale_cooldown: float = 30.0
    autoscale_idle_timeout: float = 60.0
    reconcile_tick_interval: float = 15.0
    shutdown_timeout: float = 120.0
    console: bool = True
    logging: LogConfig | bool = True


def _detect_skyward_source() -> SkywardSource:
    """Detect if skyward is installed from source (editable) or from a package registry."""
    from importlib.metadata import packages_distributions

    try:
        dist_map = packages_distributions()
        dists = dist_map.get("skyward", [])
        if not dists:
            return "local"

        from importlib.metadata import distribution

        dist = distribution(dists[0])
        direct_url = dist.read_text("direct_url.json")
        if direct_url:
            import json as _json

            data = _json.loads(direct_url)
            if data.get("dir_info", {}).get("editable", False):
                return "local"

        return "pypi"
    except Exception:
        return "pypi"


@dataclass(frozen=True, slots=True)
class PipIndex:
    """Scoped package index for uv.

    Maps specific packages to a custom index URL, generating
    ``[[tool.uv.index]]`` entries with ``explicit = true`` so that
    only the listed packages resolve from that index.

    Parameters
    ----------
    url
        Index URL (e.g. ``https://download.pytorch.org/whl/cpu``).
    packages
        Package names that should resolve from this index.

    Examples
    --------
    >>> image = Image(
    ...     pip=["torch"],
    ...     pip_indexes=[PipIndex(
    ...         url="https://download.pytorch.org/whl/cu121",
    ...         packages=["torch", "torchvision"],
    ...     )],
    ... )
    """

    url: str
    packages: list[str] | tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "packages", tuple(self.packages) if self.packages else ())


@dataclass(frozen=True, slots=True)
class Volume:
    """S3-backed volume mounted as local filesystem via FUSE.

    Parameters
    ----------
    bucket
        S3 bucket name (AWS), GCS bucket name (GCP),
        or network volume ID (RunPod).
    mount
        Absolute path where the volume appears on workers.
    prefix
        Object key prefix (subfolder within bucket).
    read_only
        Mount as read-only. Default ``True``.
    storage
        Storage backend override. ``None`` auto-detects from provider.

    Examples
    --------
    >>> compute = sky.ComputePool(
    ...     provider=sky.AWS(),
    ...     volumes=[Volume(bucket="my-data", mount="/data")],
    ... )
    """

    bucket: str
    mount: str
    prefix: str = ""
    read_only: bool = True
    storage: Storage | None = None

    def __post_init__(self) -> None:
        if not self.mount.startswith("/"):
            raise ValueError(f"mount must be absolute path, got '{self.mount}'")
        if self.mount in ("/", "/opt", "/opt/skyward", "/root", "/tmp"):
            raise ValueError(f"mount cannot be a system path: '{self.mount}'")


@dataclass(frozen=True, slots=True)
class PoolSpec:
    """Resolved pool specification — the internal, fully-normalized form.

    Created from user-facing ``Spec`` and ``Options`` objects during pool
    startup.  Carries every parameter needed to provision a cluster,
    including hardware requirements, networking, autoscaling bounds, and
    plugin configuration.

    Parameters
    ----------
    nodes
        Node count specification (fixed or autoscaling).
    accelerator
        GPU/accelerator type, or ``None`` for CPU-only.
    region
        Cloud region for instance placement.
    vcpus
        Minimum vCPUs per node.
    memory_gb
        Minimum RAM in GB per node.
    architecture
        CPU architecture filter.  ``None`` accepts any.
    allocation
        Instance lifecycle strategy (spot, on-demand, etc.).
    image
        Declarative environment specification.
    ttl
        Auto-shutdown timeout in seconds after pool exits.  ``0`` disables.
    worker
        Worker configuration (concurrency, executor backend).
    provider
        Override provider name (usually inferred from the ``Spec``).
    max_hourly_cost
        Cost cap per node per hour in USD.  ``None`` means no cap.
    ssh_timeout
        Maximum seconds to wait for an SSH connection to a node.
    ssh_retry_interval
        Seconds between SSH connection retry attempts.
    provision_retry_delay
        Seconds between provision retry attempts after a failure.
    max_provision_attempts
        Maximum number of provision attempts before giving up.
    volumes
        S3/GCS volumes to mount on workers.
    autoscale_cooldown
        Seconds between autoscaling decisions.
    autoscale_idle_timeout
        Seconds of idle before the autoscaler considers scaling down.
    reconcile_tick_interval
        Seconds between reconciler ticks (provision/drain evaluation).
    plugins
        Composable plugins applied to this pool.

    Examples
    --------
    >>> spec = PoolSpec(
    ...     nodes=Nodes(min=4),
    ...     accelerator=H100(),
    ...     region="us-east-1",
    ...     allocation="spot-if-available",
    ...     image=Image(pip=["torch"]),
    ... )
    """

    nodes: Nodes
    accelerator: Accelerator | None
    region: str
    vcpus: float | None = None
    memory_gb: float | None = None
    disk_gb: int | None = None
    architecture: Architecture | None = None
    allocation: AllocationStrategy = "spot-if-available"
    image: Image = field(default_factory=lambda: Image())
    ttl: int = 600
    worker: Worker = field(default_factory=Worker)
    provider: ProviderName | None = None
    max_hourly_cost: float | None = None
    ssh_timeout: float = 300.0
    ssh_retry_interval: float = 5.0
    provision_retry_delay: float = 10.0
    max_provision_attempts: int = 10
    volumes: tuple[Volume, ...] = ()
    autoscale_cooldown: float = 30.0
    autoscale_idle_timeout: float = 60.0
    reconcile_tick_interval: float = 15.0
    plugins: tuple[Plugin, ...] = ()

    @property
    def accelerator_name(self) -> str | None:
        """Get the canonical accelerator name for provider matching."""
        return self.accelerator.name if self.accelerator else None

    @property
    def accelerator_count(self) -> float:
        """Get the number of accelerators per node."""
        return self.accelerator.count if self.accelerator else 0

    @property
    def accelerator_memory_gb(self) -> int:
        """Get the requested VRAM per accelerator in GB, or 0 if unspecified."""
        if not self.accelerator or not self.accelerator.memory:
            return 0
        match = re.match(r"(\d+)(GB|TB)", self.accelerator.memory, re.IGNORECASE)
        if not match:
            return 0
        value = int(match.group(1))
        return value * 1024 if match.group(2).upper() == "TB" else value


def _default_metrics() -> tuple[Metric, ...]:
    from skyward.observability.metrics import Default
    return Default()


@dataclass(frozen=True, slots=True)
class Image:
    """Declarative image specification.

    Defines the environment (Python version, packages, etc.) in a
    declarative way. The ``generate_bootstrap()`` method generates idempotent
    shell scripts that work across all cloud providers.

    Parameters
    ----------
    python
        Python version to use. ``"auto"`` detects current interpreter.
    pip
        Packages to install via ``uv add``.
    pip_indexes
        Scoped package indexes. Each ``PipIndex`` maps specific packages
        to a custom index URL via uv's explicit index support.
    apt
        System packages to install via ``apt-get``.
    env
        Environment variables to export on remote workers.
    shell_vars
        Shell commands for dynamic variable capture (evaluated at bootstrap).
    includes
        Paths relative to CWD to sync to workers (dirs or ``.py`` files).
    excludes
        Glob patterns to ignore within includes (e.g., ``"__pycache__"``).
    skyward_source
        Where to install skyward from. ``"auto"`` detects editable
        installs as ``"local"``, otherwise ``"pypi"``.
    metrics
        Metrics to collect (CPU, GPU, Memory, etc.). ``None`` disables.
    bootstrap_timeout
        Maximum seconds for the bootstrap script to complete. Default ``300``.

    Examples
    --------
    >>> image = Image(
    ...     python="3.13",
    ...     pip=["torch", "transformers"],
    ...     apt=["git", "ffmpeg"],
    ...     env={"HF_TOKEN": "xxx"},
    ... )

    >>> # Disable metrics
    >>> image = Image(metrics=None)
    """

    python: str | Literal["auto"] = "auto"
    pip: list[str] | tuple[str, ...] = ()
    pip_indexes: list[PipIndex] | tuple[PipIndex, ...] = ()
    apt: list[str] | tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    shell_vars: dict[str, str] = field(default_factory=dict)
    includes: list[str] | tuple[str, ...] = ()
    excludes: list[str] | tuple[str, ...] = ()
    skyward_source: SkywardSource = "auto"
    metrics: MetricsConfig = field(default_factory=lambda: _default_metrics())
    bootstrap_timeout: int = 300

    def __post_init__(self) -> None:
        """Convert lists to tuples for immutability."""
        if self.python == "auto":
            import sys
            object.__setattr__(self, "python", f"{sys.version_info.major}.{sys.version_info.minor}")

        object.__setattr__(self, "pip", tuple(self.pip) if self.pip else ())
        object.__setattr__(self, "pip_indexes", tuple(self.pip_indexes) if self.pip_indexes else ())
        object.__setattr__(self, "apt", tuple(self.apt) if self.apt else ())
        object.__setattr__(self, "includes", tuple(self.includes) if self.includes else ())
        object.__setattr__(self, "excludes", tuple(self.excludes) if self.excludes else ())

        if self.skyward_source == "auto":
            object.__setattr__(self, "skyward_source", _detect_skyward_source())

        match self.metrics:
            case list():
                object.__setattr__(self, "metrics", tuple(self.metrics))
            case _:
                pass

    def content_hash(self) -> str:
        """Generate a deterministic hash of this image specification.

        Used by ``WarmableProvider`` implementations to detect whether
        an existing AMI/snapshot can be reused or a fresh bootstrap is
        needed.  The hash covers all fields that affect the remote
        environment (packages, env vars, Python version, etc.).

        Returns
        -------
        str
            A 12-character hex digest (SHA-256 prefix).
        """
        from skyward import __version__

        metrics_data = None
        if self.metrics:
            metrics_data = [
                {
                    "name": m.name,
                    "command": m.command,
                    "interval": m.interval,
                    "multi": m.multi,
                }
                for m in self.metrics
            ]

        content = json.dumps(
            {
                "python": self.python,
                "pip": sorted(self.pip),
                "pip_indexes": [
                    {"url": idx.url, "packages": sorted(idx.packages)}
                    for idx in self.pip_indexes
                ],
                "apt": sorted(self.apt),
                "env": dict(sorted(self.env.items())),
                "shell_vars": dict(sorted(self.shell_vars.items())),
                "includes": sorted(self.includes),
                "excludes": sorted(self.excludes),
                "skyward_source": self.skyward_source,
                "skyward_version": __version__,
                "metrics": metrics_data,
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:12]


DEFAULT_IMAGE = Image()


class PoolState:
    """Pool lifecycle states reported by the pool actor.

    Progression: ``INIT`` → ``REQUESTING`` → ``PROVISIONING``
    → ``READY`` → ``SHUTTING_DOWN`` → ``DESTROYED``.
    """

    INIT = "init"
    """Pool object created, actor system not yet started."""

    REQUESTING = "requesting"
    """Querying provider for available offers."""

    PROVISIONING = "provisioning"
    """Cloud instances being created and bootstrapped."""

    READY = "ready"
    """All nodes healthy, pool accepting tasks."""

    SHUTTING_DOWN = "shutting_down"
    """Graceful teardown in progress."""

    DESTROYED = "destroyed"
    """All instances terminated, resources released."""
