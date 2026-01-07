"""Declarative Image API for defining cloud environments.

Image is the single source of truth for environment configuration.
It generates bootstrap scripts via `bootstrap()` and provides
content-addressable hashing for AMI/snapshot caching.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal

from skyward.bootstrap import (
    Op,
    apt,
    bootstrap,
    checkpoint,
    env_export,
    install_uv,
    instance_timeout,
    systemd,
    uv_add,
    uv_init,
    wait_for_port,
)
from skyward.bootstrap.worker import rpyc_service_unit
from skyward.constants import RPYC_PORT

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
        pip: List of pip packages to install.
        pip_extra_index_url: Extra PyPI index URL.
        apt: List of apt packages to install.
        env: Environment variables to export.
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

        # Development: use local wheel via SCP
        image = Image(
            python="3.13",
            pip=["torch"],
            skyward_source="local",
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
    skyward_source: SkywardSource = "github"

    def content_hash(self) -> str:
        """Generate hash for AMI/snapshot caching.

        Two Images with the same content_hash produce identical
        environments and can share the same cached AMI/snapshot.

        Returns:
            12-character hex hash of the image specification.
        """
        content = json.dumps(
            {
                "python": self.python,
                "pip": sorted(self.pip),
                "pip_extra_index_url": self.pip_extra_index_url,
                "apt": sorted(self.apt),
                "env": dict(sorted(self.env.items())),
                "skyward_source": self.skyward_source,
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def bootstrap(
        self,
        ttl: int = 0,
        preamble: Op | None = None,
        postamble: Op | None = None,
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

        Returns:
            Complete shell script for cloud-init/user_data.
        """
        ops: list[Op | None] = [
            instance_timeout(ttl) if ttl else None,  # safety timeout ALWAYS first
            preamble,
            env_export(**self.env) if self.env else None,
            install_uv(),
            apt("python3", "curl", "git", *self.apt),  # git needed for github install
            uv_init(self.python, name="skyward-bootstrap"),  # custom name to avoid self-dependency
            uv_add(
                "cloudpickle",
                "rpyc",
                "nvidia-ml-py",  # NVIDIA GPU metrics (gracefully ignored if no GPU)
                *self.pip,
                extra_index=self.pip_extra_index_url,
            ),
            checkpoint(".step_pip"),
        ]

        # Install skyward if not using local wheel
        if self.skyward_source == "github":
            ops.append(uv_add("git+https://github.com/gabfssilva/skyward.git"))
            ops.append(checkpoint(".step_wheel"))
        elif self.skyward_source == "pypi":
            ops.append(uv_add("skyward"))
            ops.append(checkpoint(".step_wheel"))
        # "local" mode: skyward installed via SCP after bootstrap

        # Start RPyC service if skyward is installed via user-data
        if self.skyward_source != "local":
            ops.append(systemd("skyward-rpyc", rpyc_service_unit(env=self.env)))
            ops.append(wait_for_port(RPYC_PORT))
            ops.append(checkpoint(".step_server"))

        ops.append(postamble)
        ops.append(checkpoint(".ready"))

        return bootstrap(*ops)


# Default image for when none is specified
DEFAULT_IMAGE = Image()
