"""Common utilities shared between providers (wheel building, etc.)."""

from __future__ import annotations

import fnmatch
import io
import subprocess
import tarfile
import tempfile
from pathlib import Path

from skyward.infra.ssh import SSHTransport
from skyward.observability.logger import logger
from skyward.providers.bootstrap.compose import SKYWARD_DIR


async def detect_network_interface(transport: SSHTransport) -> str:
    """Detect the default network interface on a remote node via SSH.

    Runs `ip -o route show default` and parses the `dev` field.
    Works regardless of AMI/distro (Ubuntu ens5, Amazon Linux eth0, etc.).
    """
    exit_code, stdout, _ = await transport.run("ip -o route show default", timeout=10.0)
    if exit_code != 0 or "dev " not in stdout:
        return ""

    return stdout.split("dev ")[1].split()[0]


def build_wheel() -> Path:
    """Build skyward wheel locally into a fresh temp directory."""
    logger.debug("Building skyward wheel locally...")
    skyward_dir = Path(__file__).parent.parent
    project_dir = skyward_dir.parent

    build_dir = tempfile.mkdtemp(prefix="skyward-wheel-")

    result = subprocess.run(
        ["uv", "build", "--wheel", "-o", build_dir],
        cwd=project_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("Failed to build wheel: {err}", err=result.stderr)
        raise RuntimeError(f"Failed to build wheel: {result.stderr}")

    wheel_path = next(Path(build_dir).glob("*.whl"))
    logger.info("Built wheel: {name}", name=wheel_path.name)
    return wheel_path


def _build_wheel_install_script(wheel_name: str) -> str:
    """Build wheel installation script.

    Installs the skyward wheel into the venv. Service startup (Casty)
    is handled by bootstrap, not this script.

    Parameters
    ----------
    wheel_name
        Name of the wheel file to install.

    Returns
    -------
    str
        Shell script string.
    """
    script = f"""#!/bin/bash
set -e

# Move wheel from /tmp to SKYWARD_DIR (uploaded to /tmp for permission reasons)
mv /tmp/{wheel_name} {SKYWARD_DIR}/{wheel_name} 2>/dev/null || true

# Find uv
UV_PATH=$(which uv 2>/dev/null || find /root /home -name uv -type f 2>/dev/null | head -1)
if [ -z "$UV_PATH" ]; then
    UV_PATH="/root/.local/bin/uv"
fi

# Install wheel
cd {SKYWARD_DIR}
$UV_PATH pip install {SKYWARD_DIR}/{wheel_name}

# Verify installation
{SKYWARD_DIR}/.venv/bin/python -c 'import skyward; print(skyward.__file__)'

# Verify Casty can be imported
{SKYWARD_DIR}/.venv/bin/python -c 'import casty; print("Casty OK")'
"""
    return script


_DEFAULT_EXCLUDES = (
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".git",
    ".venv",
    "node_modules",
    "*.egg-info",
)


def build_user_code_tarball(
    includes: tuple[str, ...],
    excludes: tuple[str, ...] = (),
    project_root: Path | None = None,
) -> bytes:
    """Build a tar.gz archive of user code paths for syncing to workers.

    Parameters
    ----------
    includes
        Paths relative to project_root (dirs or .py files).
    excludes
        Additional glob patterns to ignore.
    project_root
        Root directory to resolve paths from. Defaults to CWD.

    Returns
    -------
    bytes
        Compressed tar bytes ready for upload.
    """
    root = project_root or Path.cwd()
    all_excludes = _DEFAULT_EXCLUDES + tuple(excludes)

    def _is_excluded(path: Path) -> bool:
        return any(
            fnmatch.fnmatch(part, pattern)
            for part in path.parts
            for pattern in all_excludes
        )

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for include_path in includes:
            source = root / include_path
            if not source.exists():
                logger.warning(
                    "Include path does not exist, skipping: {path}",
                    path=include_path,
                )
                continue

            if source.is_file():
                tar.add(str(source), arcname=include_path)
            elif source.is_dir():
                for file in source.rglob("*"):
                    if file.is_file():
                        rel = file.relative_to(root)
                        if not _is_excluded(rel):
                            tar.add(str(file), arcname=str(rel))

    tarball = buf.getvalue()
    logger.info(
        "Built user code tarball: {size:.1f} KB from {n} includes",
        size=len(tarball) / 1024,
        n=len(includes),
    )
    return tarball
