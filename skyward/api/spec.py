"""Specification dataclasses for pool and image configuration.

These are the immutable configuration objects that define what
the user wants. Components use these specs to provision resources.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from skyward.observability.metrics import Default as DefaultMetrics
from skyward.observability.metrics import MetricsConfig
from skyward.providers.bootstrap import (
    Op,
    apt,
    bootstrap,
    emit_bootstrap_complete,
    env_export,
    install_uv,
    instance_timeout,
    phase,
    phase_simple,
    shell_vars,
    start_metrics,
    uv_add,
    uv_configure_indexes,
    uv_init,
)

if TYPE_CHECKING:
    from skyward.accelerators import Accelerator
    from skyward.actors.messages import ProviderName
    from skyward.api.provider import ProviderConfig
    from skyward.plugins.plugin import Plugin
    from skyward.storage import Storage


type AllocationStrategy = Literal[
    "spot",
    "on-demand",
    "spot-if-available",
    "cheapest",
]
"""Instance lifecycle strategy.

- ``"spot"`` — spot/preemptible only (cheapest, may be interrupted).
- ``"on-demand"`` — on-demand only (reliable, higher cost).
- ``"spot-if-available"`` — try spot first, fall back to on-demand.
- ``"cheapest"`` — compare spot and on-demand, pick lowest price.
"""

type Architecture = Literal["x86_64", "arm64"]
"""CPU architecture filter."""

type SelectionStrategy = Literal["first", "cheapest"]
"""Multi-spec selection strategy.

- ``"first"`` — use the first spec that has available offers.
- ``"cheapest"`` — compare all specs, pick the lowest price.
"""

type SkywardSource = Literal["auto", "local", "github", "pypi"]
"""Skyward installation source for remote workers."""

type WorkerExecutor = Literal["auto", "thread", "process"]
"""Worker execution backend.

- ``"auto"`` — resolves to ``"thread"`` (default).
- ``"thread"`` — ThreadPoolExecutor. Supports streaming, distributed
  collections without IPC, and has lower overhead.
- ``"process"`` — ProcessPoolExecutor. Useful for CPU-bound pure-Python
  workloads that benefit from bypassing the GIL.
"""


@dataclass(frozen=True, slots=True)
class Worker:
    """Worker configuration per node.

    Parameters
    ----------
    concurrency
        Number of concurrent task slots per node. Default ``1``.
    executor
        Execution backend — ``"auto"`` (default) resolves to ``"thread"``.
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
    """Hardware specification for multi-provider fallback chains.

    Pass one or more ``Spec`` objects to ``ComputePool`` to define
    fallback preferences across providers. The pool selects the best
    match based on the ``selection`` strategy.

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
    region
        Cloud region (provider-specific).
    max_hourly_cost
        Cost cap per node per hour in USD.
    ttl
        Auto-shutdown timeout in seconds after pool exits. ``0`` disables.

    Examples
    --------
    >>> with sky.ComputePool(
    ...     sky.Spec(provider=sky.VastAI(), accelerator="A100"),
    ...     sky.Spec(provider=sky.AWS(), accelerator="A100"),
    ...     selection="cheapest",
    ... ) as pool:
    ...     result = train(data) >> pool
    """
    provider: ProviderConfig
    accelerator: Accelerator | None = None
    nodes: int | tuple[int, int] = 1
    vcpus: float | None = None
    memory_gb: float | None = None
    architecture: Architecture | None = None
    allocation: AllocationStrategy = "spot-if-available"
    region: str | None = None
    max_hourly_cost: float | None = None
    ttl: int = 600


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
    >>> pool = sky.ComputePool(
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
    """Internal pool specification — resolved from user-facing ``ComputePool`` args.

    Defines the cluster configuration including number of nodes,
    hardware requirements, and instance allocation strategy.

    Parameters
    ----------
    nodes
        Number of nodes in the cluster.
    accelerator
        GPU/accelerator type, or ``None`` for CPU-only.
    region
        Cloud region for instances.
    vcpus
        Minimum vCPUs per node.
    memory_gb
        Minimum RAM in GB per node.
    architecture
        CPU architecture filter. ``None`` accepts any.
    allocation
        Instance lifecycle strategy.
    image
        Environment specification.
    ttl
        Auto-shutdown timeout in seconds. ``0`` disables.
    worker
        Worker configuration (concurrency, executor).
    provider
        Override provider name (usually inferred from context).
    max_hourly_cost
        Cost cap per node per hour in USD.

    Examples
    --------
    >>> spec = PoolSpec(
    ...     nodes=4,
    ...     accelerator=H100(),
    ...     region="us-east-1",
    ...     allocation="spot-if-available",
    ...     image=Image(pip=["torch"]),
    ... )
    """

    nodes: int
    accelerator: Accelerator | None
    region: str
    vcpus: float | None = None
    memory_gb: float | None = None
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
    min_nodes: int | None = None
    max_nodes: int | None = None
    autoscale_cooldown: float = 30.0
    autoscale_idle_timeout: float = 60.0
    reconcile_tick_interval: float = 15.0
    plugins: tuple[Plugin, ...] = ()

    def __post_init__(self) -> None:
        if self.nodes < 1:
            raise ValueError(f"nodes must be >= 1, got {self.nodes}")
        if self.min_nodes is not None and self.max_nodes is not None:
            if self.min_nodes < 1:
                raise ValueError(f"min_nodes must be >= 1, got {self.min_nodes}")
            if self.max_nodes < self.min_nodes:
                raise ValueError(
                    f"max_nodes ({self.max_nodes}) must be >= min_nodes ({self.min_nodes})"
                )

    @property
    def accelerator_name(self) -> str | None:
        """Get the canonical accelerator name for provider matching."""
        return self.accelerator.name if self.accelerator else None

    @property
    def accelerator_count(self) -> int:
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

    @property
    def auto_scaling(self) -> bool:
        return self.min_nodes is not None and self.max_nodes is not None


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
    metrics: MetricsConfig = field(default_factory=lambda: DefaultMetrics())
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
        """Generate hash for AMI/snapshot caching."""
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

    def generate_bootstrap(
        self,
        ttl: int = 0,
        shutdown_command: str = "shutdown -h now",
        preamble: Op | None = None,
        postamble: Op | None = None,
    ) -> str:
        """Generate bootstrap script for cloud-init/user_data."""
        ops: list[Op | None] = [
            instance_timeout(ttl, shutdown_command=shutdown_command) if ttl else None,
            preamble,
        ]

        if self.shell_vars or self.env:
            env_ops: list[Op] = []
            if self.shell_vars:
                env_ops.append(shell_vars(**self.shell_vars))
            if self.env:
                env_ops.append(env_export(**self.env))
            ops.append(phase_simple("env", *env_ops))

        ops.extend([
            phase("apt", apt("curl", "git", *self.apt)),
            phase_simple(
                "uv",
                install_uv(),
                uv_init(self.python, name="skyward-bootstrap"),
                uv_configure_indexes(self.pip_indexes),
            ),
            phase("deps", uv_add("cloudpickle", "lz4", *self.pip)),
        ])

        match self.skyward_source:
            case "github":
                ops.append(phase("skyward", uv_add("git+https://github.com/gabfssilva/skyward.git")))
            case "pypi":
                ops.append(phase("skyward", uv_add("skyward")))

        ops.append(postamble)
        ops.append(emit_bootstrap_complete())
        ops.append(start_metrics())

        return bootstrap(*ops, metrics=self.metrics)


DEFAULT_IMAGE = Image()


class PoolState:
    """Pool lifecycle states."""

    INIT = "init"
    REQUESTING = "requesting"
    PROVISIONING = "provisioning"
    READY = "ready"
    SHUTTING_DOWN = "shutting_down"
    DESTROYED = "destroyed"
