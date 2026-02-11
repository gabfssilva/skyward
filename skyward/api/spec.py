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


type AllocationStrategy = Literal[
    "spot",
    "on-demand",
    "spot-if-available",
    "cheapest",
]

type Architecture = Literal["x86_64", "arm64"]

type SkywardSource = Literal["local", "github", "pypi"]


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
    vcpus: int | None = None
    memory_gb: int | None = None
    architecture: Architecture | None = None
    allocation: AllocationStrategy = "spot-if-available"
    image: Image = field(default_factory=lambda: Image())
    ttl: int = 600
    concurrency: int = 1
    provider: ProviderName | None = None
    max_hourly_cost: float | None = None

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
        skyward_source: Where to install skyward from.
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

    python: str = "3.13"
    pip: list[str] | tuple[str, ...] = ()
    pip_extra_index_url: str | None = None
    apt: list[str] | tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    shell_vars: dict[str, str] = field(default_factory=dict)
    skyward_source: SkywardSource = "github"
    metrics: MetricsConfig = field(default_factory=lambda: DefaultMetrics())

    def __post_init__(self) -> None:
        """Convert lists to tuples for immutability."""
        object.__setattr__(self, "pip", tuple(self.pip) if self.pip else ())
        object.__setattr__(self, "apt", tuple(self.apt) if self.apt else ())

        match self.metrics:
            case list():
                object.__setattr__(self, "metrics", tuple(self.metrics))
            case _:
                pass

    def content_hash(self) -> str:
        """Generate hash for AMI/snapshot caching."""
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
                "skyward_source": self.skyward_source,
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
