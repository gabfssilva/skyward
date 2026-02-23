"""Specification dataclasses for pool and image configuration.

These are the immutable configuration objects that define what
the user wants. Components use these specs to provision resources.
"""

from __future__ import annotations

import hashlib
import json
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
    uv_init,
)

if TYPE_CHECKING:
    from skyward.accelerators import Accelerator
    from skyward.actors.messages import ProviderName
    from skyward.api.provider import ProviderConfig


type AllocationStrategy = Literal[
    "spot",
    "on-demand",
    "spot-if-available",
    "cheapest",
]

type Architecture = Literal["x86_64", "arm64"]

type SelectionStrategy = Literal["first", "cheapest"]

type SkywardSource = Literal["auto", "local", "github", "pypi"]


type WorkerExecutor = Literal["auto", "thread", "process"]


@dataclass(frozen=True, slots=True)
class Worker:
    """Worker configuration per node.

    Args:
        concurrency: Number of concurrent task slots per node.
        executor: Execution backend — "auto" (default) resolves to "thread".
            Use "process" (ProcessPoolExecutor) explicitly for CPU-bound pure-Python
            workloads that benefit from bypassing the GIL. Thread executor supports
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
    """User-facing hardware preference for ComputePool fallback chains."""
    provider: ProviderConfig
    accelerator: Accelerator | str | None = None
    nodes: int = 1
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
class Volume:
    """S3-backed volume mounted as local filesystem via FUSE.

    Parameters
    ----------
    bucket : str
        S3 bucket name (AWS), GCS bucket name (GCP),
        or network volume ID (RunPod).
    mount : str
        Absolute path where the volume appears on workers.
    prefix : str
        Object key prefix (subfolder within bucket).
    read_only : bool
        Mount as read-only. Default True.
    """

    bucket: str
    mount: str
    prefix: str = ""
    read_only: bool = True

    def __post_init__(self) -> None:
        if not self.mount.startswith("/"):
            raise ValueError(f"mount must be absolute path, got '{self.mount}'")
        if self.mount in ("/", "/opt", "/opt/skyward", "/root", "/tmp"):
            raise ValueError(f"mount cannot be a system path: '{self.mount}'")


@dataclass(frozen=True, slots=True)
class PoolSpec:
    """Pool specification - what the user wants.

    Defines the cluster configuration including number of nodes,
    hardware requirements, and instance allocation strategy.

    Args:
        nodes: Number of nodes in the cluster.
        accelerator: GPU/accelerator type - either a string (e.g., "A100", "H100")
            or an Accelerator instance from skyward.accelerators.
        region: Cloud region for instances.
        vcpus: Minimum vCPUs per node.
        memory_gb: Minimum memory in GB per node.
        architecture: CPU architecture ("x86_64" or "arm64"), or None for cheapest.
        allocation: Spot/on-demand strategy.
        image: Environment specification.
        ttl: Auto-shutdown timeout in seconds (0 = disabled).
        provider: Override provider (usually inferred from context).

    Example:
        >>> spec = PoolSpec(
        ...     nodes=4,
        ...     accelerator="H100",
        ...     region="us-east-1",
        ...     allocation="spot-if-available",
        ...     image=Image(pip=["torch"]),
        ... )

        >>> from skyward.accelerators import H100
        >>> spec = PoolSpec(
        ...     nodes=4,
        ...     accelerator=H100(count=8),
        ...     region="us-east-1",
        ... )
    """

    nodes: int
    accelerator: Accelerator | str | None
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

    def __post_init__(self) -> None:
        if self.nodes < 1:
            raise ValueError(f"nodes must be >= 1, got {self.nodes}")

    @property
    def accelerator_name(self) -> str | None:
        """Get the canonical accelerator name for provider matching."""
        match self.accelerator:
            case None:
                return None
            case str(name):
                return name
            case accel:
                return accel.name

    @property
    def accelerator_count(self) -> int:
        """Get the number of accelerators per node."""
        match self.accelerator:
            case None:
                return 0
            case str():
                return 1
            case accel:
                return accel.count


@dataclass(frozen=True, slots=True)
class Image:
    """Declarative image specification.

    Defines the environment (Python version, packages, etc.) in a
    declarative way. The bootstrap() method generates idempotent
    shell scripts that work across all cloud providers.

    Args:
        python: Python version to use.
        pip: List of pip packages to install.
        pip_extra_index_url: Extra PyPI index URL.
        apt: List of apt packages to install.
        env: Environment variables to export.
        shell_vars: Shell commands for dynamic variable capture.
        includes: Paths relative to CWD to sync to workers (dirs or .py files).
        excludes: Glob patterns to ignore within includes (e.g., "__pycache__", "*.pyc").
        skyward_source: Where to install skyward from. "auto" detects editable
            installs as "local", otherwise "pypi".
        metrics: Metrics to collect (CPU, GPU, Memory, etc.). Use None to disable.

    Example:
        image = Image(
            python="3.13",
            pip=["torch", "transformers"],
            apt=["git", "ffmpeg"],
            env={"HF_TOKEN": "xxx"},
        )

        # Custom metrics
        from skyward.spec.metrics import CPU, GPU
        image = Image(
            pip=["torch"],
            metrics=[CPU(interval=0.5), GPU()],
        )

        # Disable metrics
        image = Image(metrics=None)

        # Generate bootstrap script
        script = image.bootstrap(ttl=3600)
    """

    python: str | Literal["auto"] = "auto"
    pip: list[str] | tuple[str, ...] = ()
    pip_extra_index_url: str | None = None
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
                "pip_extra_index_url": self.pip_extra_index_url,
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
            phase_simple("uv", install_uv(), uv_init(self.python, name="skyward-bootstrap")),
            phase(
                "deps",
                uv_add(
                    "cloudpickle",
                    "lz4",
                    *self.pip,
                    extra_index=self.pip_extra_index_url,
                ),
            ),
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
