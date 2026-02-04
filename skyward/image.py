"""Declarative Image API for v2.

Image is the specification for instance environments. It generates
bootstrap scripts and provides content-addressable hashing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal

from .metrics import MetricsConfig, Default as DefaultMetrics

from .bootstrap import (
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


# Skyward installation source
type SkywardSource = Literal["local", "github", "pypi"]

# Ray client port
RAY_CLIENT_PORT = 10001


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
        # Use object.__setattr__ since the dataclass is frozen
        object.__setattr__(self, "pip", tuple(self.pip) if self.pip else ())
        object.__setattr__(self, "apt", tuple(self.apt) if self.apt else ())
        # Convert metrics list to tuple if needed
        if isinstance(self.metrics, list):
            object.__setattr__(self, "metrics", tuple(self.metrics))

    def content_hash(self) -> str:
        """Generate hash for AMI/snapshot caching.

        Two Images with the same content_hash produce identical
        environments and can share the same cached AMI/snapshot.

        Returns:
            12-character hex hash of the image specification.
        """
        # Serialize metrics for hashing
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
        preamble: Op | None = None,
        postamble: Op | None = None,
        use_systemd: bool = True,  # noqa: ARG002 - kept for compatibility
    ) -> str:
        """Generate bootstrap script for cloud-init/user_data.

        The script is idempotent: if AMI already has deps installed,
        uv add verifies and skips quickly.

        Args:
            ttl: Auto-shutdown timeout in seconds (0 = disabled).
            preamble: Op to execute first.
            postamble: Op to execute last.
            use_systemd: Unused (kept for compatibility).

        Returns:
            Complete shell script for cloud-init.
        """
        ops: list[Op | None] = [
            instance_timeout(ttl) if ttl else None,
            preamble,
        ]

        # Environment setup (only if we have vars or env)
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

        # Install skyward (skip for local - wheel uploaded separately)
        if self.skyward_source == "github":
            ops.append(phase("skyward", uv_add("git+https://github.com/gabfssilva/skyward.git")))
        elif self.skyward_source == "pypi":
            ops.append(phase("skyward", uv_add("skyward")))

        ops.append(postamble)
        ops.append(emit_bootstrap_complete())
        ops.append(start_metrics())

        return bootstrap(*ops, metrics=self.metrics)


# Default image
DEFAULT_IMAGE = Image()


__all__ = [
    "Image",
    "SkywardSource",
    "DEFAULT_IMAGE",
    "RAY_CLIENT_PORT",
]
