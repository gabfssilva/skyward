"""Build and upload skyward wheel to S3."""

from __future__ import annotations

import hashlib
import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skyward.providers.aws.s3 import S3ObjectStore

logger = logging.getLogger("skyward.aws.wheel")


def _find_project_root() -> Path:
    """Find project root by looking for pyproject.toml."""
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find project root (no pyproject.toml found)")


def ensure_skyward_wheel(store: S3ObjectStore) -> str:
    """Build and upload skyward wheel to S3 if needed.

    Args:
        store: S3 object store instance.

    Returns:
        S3 key for the skyward wheel.
    """
    project_dir = _find_project_root()
    skyward_dir = project_dir / "skyward"

    # Hash the skyward source code
    source_hash = hashlib.sha256()
    for py_file in sorted(skyward_dir.rglob("*.py")):
        source_hash.update(py_file.read_bytes())
    content_hash = source_hash.hexdigest()[:12]

    wheel_name = "skyward-0.1.0-py3-none-any.whl"
    wheel_key = f"wheel/{content_hash}/{wheel_name}"

    if not store.exists(wheel_key):
        logger.info(f"Building skyward wheel (hash={content_hash})...")
        build_dir = Path("/tmp/skyward-wheel")
        build_dir.mkdir(exist_ok=True)

        result = subprocess.run(
            ["uv", "build", "--wheel", "-o", str(build_dir)],
            cwd=project_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to build skyward wheel: {result.stderr}")

        wheel_path = next(build_dir.glob("*.whl"))
        logger.info("Uploading skyward wheel to S3...")
        store.put(wheel_key, wheel_path.read_bytes())

    return wheel_key
