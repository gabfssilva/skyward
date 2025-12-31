"""Declarative Image API for defining cloud environments.

Image is the single source of truth for environment configuration.
It generates bootstrap scripts via `bootstrap()` and provides
content-addressable hashing for AMI/snapshot caching.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from skyward.bootstrap import (
    Op,
    apt,
    bootstrap,
    checkpoint,
    env_export,
    install_uv,
    instance_timeout,
    uv_add,
    uv_init,
)


@dataclass(frozen=True)
class Image:
    """Declarative image specification - única fonte para bootstrap.

    Defines the environment (Python version, packages, etc.) in a
    declarative way. The bootstrap() method generates idempotent
    shell scripts that work across all cloud providers.

    Example:
        image = Image(
            python="3.13",
            pip=["torch", "transformers"],
            apt=["git", "ffmpeg"],
            env={"HF_TOKEN": "xxx"},
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

        Args:
            ttl: Auto-shutdown in seconds (0 = disabled).
            preamble: Op to execute first (e.g., ssm_restart() on AWS).
            postamble: Op to execute last (e.g., volume mounts).

        Returns:
            Complete shell script for cloud-init/user_data.
        """
        return bootstrap(
            preamble,
            instance_timeout(ttl) if ttl else None,
            env_export(**self.env) if self.env else None,
            install_uv(),
            apt("python3", "curl", *self.apt),
            uv_init(self.python),
            uv_add("cloudpickle", "rpyc", *self.pip, extra_index=self.pip_extra_index_url),
            # NOTE: systemd service is created AFTER wheel installation via SCP
            # because python -m skyward.rpc requires the skyward package
            postamble,
            checkpoint(".ready"),
        )


# Default image for when none is specified
DEFAULT_IMAGE = Image()
