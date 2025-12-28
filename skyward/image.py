"""Declarative Image API for defining container environments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field, model_validator


class Image(BaseModel):
    """Declarative container image definition.

    Examples:
        # Simple Python image with packages
        image = Image(
            python="3.13",
            pip=["numpy", "pandas"],
            apt=["ffmpeg"],
        )

        # Custom base image
        image = Image(
            base="ubuntu:22.04",
            pip=["numpy"],
        )

        # From existing Dockerfile
        image = Image(dockerfile="./Dockerfile")

        # Pre-built image (no building needed)
        image = Image(base="my-registry/my-image:latest")

        # Copy local files into the container
        image = Image(
            python="3.13",
            copy_local={"./models": "/app/models"},
        )
    """

    # Base image specification (mutually exclusive options)
    python: str | None = Field(default=None, description="Python version (e.g., '3.13')")
    base: str | None = Field(default=None, description="Base Docker image")
    dockerfile: str | None = Field(default=None, description="Path to Dockerfile")

    # Package installation
    pip: list[str] = Field(default_factory=list, description="Pip packages to install")
    pip_extra_index_url: str | None = Field(default=None, description="Custom pip index URL")
    apt: list[str] = Field(default_factory=list, description="Apt packages to install")

    # Environment and files
    env: dict[str, str] = Field(default_factory=dict, description="Environment variables")
    copy_local: dict[str, str] = Field(default_factory=dict, description="Files to copy")

    # Custom commands
    run: list[str] = Field(default_factory=list, description="Custom RUN commands")
    workdir: str = Field(default="/app", description="Working directory")

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_base_specification(self) -> Self:
        """Ensure exactly one base specification is provided."""
        specs = [self.python, self.base, self.dockerfile]
        provided = sum(1 for s in specs if s is not None)

        if provided == 0:
            # Default to Python 3.13
            object.__setattr__(self, "python", "3.13")
        elif provided > 1:
            raise ValueError("Only one of 'python', 'base', or 'dockerfile' can be specified")

        return self

    def get_base_image(self) -> str:
        """Get the base Docker image name."""
        if self.python:
            return f"python:{self.python}-slim-bookworm"
        elif self.base:
            return self.base
        elif self.dockerfile:
            # Parse FROM instruction from Dockerfile
            dockerfile_path = Path(self.dockerfile)
            if not dockerfile_path.exists():
                raise FileNotFoundError(f"Dockerfile not found: {self.dockerfile}")

            content = dockerfile_path.read_text()
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.upper().startswith("FROM"):
                    return stripped.split()[1]

            raise ValueError(f"No FROM instruction found in Dockerfile: {self.dockerfile}")
        else:
            return "python:3.13-slim-bookworm"

    def to_dockerfile(self) -> str:
        """Generate a Dockerfile from this image definition."""
        if self.dockerfile:
            # Use existing Dockerfile content
            return Path(self.dockerfile).read_text()

        lines: list[str] = []

        # FROM
        lines.append(f"FROM {self.get_base_image()}")

        # WORKDIR
        lines.append(f"WORKDIR {self.workdir}")

        # APT packages
        if self.apt:
            apt_cmd = (
                "apt-get update && "
                "apt-get install -y --no-install-recommends "
                f"{' '.join(self.apt)} && "
                "rm -rf /var/lib/apt/lists/*"
            )
            lines.append(f"RUN {apt_cmd}")

        # PIP packages
        if self.pip:
            lines.append(f"RUN pip install --no-cache-dir {' '.join(self.pip)}")

        # Environment variables
        for key, value in self.env.items():
            lines.append(f"ENV {key}={value}")

        # Copy files
        for src, dest in self.copy_local.items():
            lines.append(f"COPY {src} {dest}")

        # Custom RUN commands
        for cmd in self.run:
            lines.append(f"RUN {cmd}")

        return "\n".join(lines)

    def content_hash(self) -> str:
        """Generate a hash of the image definition for caching.

        Includes the skyward source code to ensure image is rebuilt when
        the package changes (since skyward is now installed in the image).
        """
        # Hash all .py files in skyward package
        skyward_dir = Path(__file__).parent
        source_hash = hashlib.sha256()
        for py_file in sorted(skyward_dir.rglob("*.py")):
            source_hash.update(py_file.read_bytes())

        content = json.dumps(
            {
                "python": self.python,
                "base": self.base,
                "dockerfile": self.dockerfile,
                "pip": sorted(self.pip),
                "apt": sorted(self.apt),
                "env": self.env,
                "copy_local": self.copy_local,
                "run": self.run,
                "workdir": self.workdir,
                "skyward_source": source_hash.hexdigest(),
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def requirements_hash(self) -> str:
        """Generate a hash of runtime requirements for UV-based execution.

        This hash is used to match stopped EC2 instances with the same
        dependencies, enabling fast instance reuse without Docker.

        The hash includes:
        - Python version
        - pip packages and index URL
        - apt packages
        - environment variables
        """
        content = json.dumps(
            {
                "python": self.python or "3.13",
                "pip": sorted(self.pip),
                "pip_extra_index_url": self.pip_extra_index_url,
                "apt": sorted(self.apt),
                "env": self.env,
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def to_requirements_txt(self) -> str:
        """Generate requirements.txt content for UV installation.

        Returns:
            Requirements file content with all pip packages.
        """
        lines: list[str] = []

        # Add extra index URL if specified (keeps PyPI as primary)
        if self.pip_extra_index_url:
            lines.append(f"--extra-index-url {self.pip_extra_index_url}")

        # User packages + runtime dependencies
        packages = set(self.pip)
        packages.update(["cloudpickle", "boto3", "pydantic"])
        lines.extend(sorted(packages))

        return "\n".join(lines)

    def tag(self) -> str:
        """Generate a Docker tag for this image."""
        return f"skyward:{self.content_hash()}"

    @property
    def local_files(self) -> list[tuple[str, str]]:
        """Get list of local files to copy (for build context)."""
        return list(self.copy_local.items())


# Default image for functions without a custom image
DEFAULT_IMAGE = Image(python="3.13")
