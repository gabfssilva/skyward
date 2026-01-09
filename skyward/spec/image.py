"""Declarative Image API for defining cloud environments.

Image is the single source of truth for environment configuration.
It generates bootstrap scripts via `bootstrap()` and provides
content-addressable hashing for AMI/snapshot caching.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from skyward.bootstrap import (
    Op,
    apt,
    bootstrap,
    emit_bootstrap_complete,
    env_export,
    install_uv,
    instance_timeout,
    nohup_service,
    phase,
    phase_simple,
    shell_vars,
    start_metrics,
    systemd,
    uv_add,
    uv_init,
    wait_for_port,
)
from skyward.bootstrap.worker import rpyc_service_unit
from skyward.core.constants import RPYC_PORT

if TYPE_CHECKING:
    from skyward.spec.metrics import Metric

# Type alias for metrics configuration
type MetricsConfig = tuple["Metric", ...] | list["Metric"] | None

# Skyward installation source
type SkywardSource = Literal["local", "github", "pypi"]


@dataclass(frozen=True)
class Image:
    """Declarative image specification - única fonte para bootstrap.

    Defines the environment (Python version, packages, etc.) in a
    declarative way. The bootstrap() method generates idempotent
    shell scripts that work across all cloud providers.

    Args:
        python: Python version to use (default: "3.13").
        pip: List of pip packages to install. Supports shell variable
            interpolation via ${VAR} syntax (see shell_vars).
        pip_extra_index_url: Extra PyPI index URL.
        apt: List of apt packages to install. Supports ${VAR} interpolation.
        env: Environment variables to export. Supports ${VAR} interpolation.
        shell_vars: Shell commands to execute remotely, with their output
            captured into variables. These variables can be interpolated
            into pip, apt, and env using ${VAR} syntax.
        skyward_source: Where to install skyward from:
            - "local": Build and SCP wheel from local machine (dev mode)
            - "github": Install from GitHub repository (default)
            - "pypi": Install from PyPI (when available)

    Example:
        # Production: install skyward from GitHub
        image = Image(
            python="3.13",
            pip=["torch", "transformers"],
            apt=["git", "ffmpeg"],
            env={"HF_TOKEN": "xxx"},
        )

        # Dynamic versions using shell_vars
        image = Image(
            pip=["torch==${CUDA_VER}"],
            shell_vars={
                "CUDA_VER": "nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1",
            },
        )

        # Generate bootstrap script
        script = image.bootstrap(ttl=3600)

        # Cache key for AMI/snapshot
        hash = image.content_hash()
    """

    python: str = "3.13"
    pip: list[str] = field(default_factory=list)
    pip_extra_index_url: str | None = None
    apt: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    shell_vars: dict[str, str] = field(default_factory=dict)
    skyward_source: SkywardSource = "github"
    metrics: MetricsConfig = None  # None means use Default()

    def content_hash(self) -> str:
        """Generate hash for AMI/snapshot caching.

        Two Images with the same content_hash produce identical
        environments and can share the same cached AMI/snapshot.

        Returns:
            12-character hex hash of the image specification.
        """
        # Serialize metrics config for hashing
        metrics_key: list[tuple[str, str, float, bool]] | None = None
        if self.metrics is not None:
            metrics_key = [
                (m.name, m.command, m.interval, m.multi)
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
                "metrics": metrics_key,
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def bootstrap(
        self,
        ttl: int = 0,
        preamble: Op | None = None,
        postamble: Op | None = None,
        use_systemd: bool = True,
    ) -> str:
        """Generate bootstrap script - identical for all clouds.

        The script is idempotent: if AMI already has deps installed,
        uv add verifies and skips quickly (~seconds).

        When skyward_source is "github" or "pypi", the bootstrap script
        will install skyward and start the RPyC service automatically.
        When "local", skyward is installed via SCP after the bootstrap.

        Args:
            ttl: Auto-shutdown in seconds (0 = disabled).
            preamble: Op to execute first (e.g., ssm_restart() on AWS).
            postamble: Op to execute last (e.g., volume mounts).
            use_systemd: If True, use systemd to manage RPyC service.
                If False, use nohup (for Docker containers without systemd).

        Returns:
            Complete shell script for cloud-init/user_data.
        """
        ops: list[Op | None] = [
            instance_timeout(ttl) if ttl else None,  # safety timeout ALWAYS first
            preamble,
            # Environment setup (fast, no streaming needed)
            phase_simple(
                "env",
                shell_vars(**self.shell_vars) if self.shell_vars else None,
                env_export(**self.env) if self.env else None,
            ) if self.shell_vars or self.env else None,
            phase("apt", apt("curl", "git", *self.apt)),
            phase_simple("uv", install_uv(), uv_init(self.python, name="skyward-bootstrap")),
            phase(
                "deps",
                uv_add(
                    "cloudpickle",
                    "rpyc",
                    *self.pip,
                    extra_index=self.pip_extra_index_url,
                ),
            ),
        ]

        # Install skyward if not using local wheel
        if self.skyward_source == "github":
            ops.append(phase("skyward", uv_add("git+https://github.com/gabfssilva/skyward.git")))
        elif self.skyward_source == "pypi":
            ops.append(phase("skyward", uv_add("skyward")))
        # "local" mode: skyward installed via SCP after bootstrap

        # Start RPyC service if skyward is installed via user-data
        if self.skyward_source != "local":
            if use_systemd:
                ops.append(
                    phase_simple(
                        "server",
                        systemd("skyward-rpyc", rpyc_service_unit(env=self.env)),
                        wait_for_port(RPYC_PORT),
                    )
                )
            else:
                # Docker containers without systemd - use nohup
                ops.append(
                    phase_simple(
                        "server",
                        nohup_service(
                            "skyward-rpyc",
                            ".venv/bin/python -m skyward.rpc",
                            env=self.env,
                        ),
                        wait_for_port(RPYC_PORT),
                    )
                )

        ops.append(postamble)
        ops.append(emit_bootstrap_complete())
        ops.append(start_metrics())  # Start metrics daemon after bootstrap

        # Determine effective metrics config
        effective_metrics = self.metrics
        if effective_metrics is None:
            # Default metrics when not specified
            from skyward.spec.metrics import Default
            effective_metrics = Default()

        return bootstrap(*ops, metrics=effective_metrics)


# Default image for when none is specified
DEFAULT_IMAGE = Image()
