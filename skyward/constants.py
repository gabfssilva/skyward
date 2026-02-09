"""Centralized constants and enums for Skyward.

All magic strings, paths, and configuration constants are defined here
to ensure consistency and enable type-safe usage throughout the codebase.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

# =============================================================================
# EC2 Instance States
# =============================================================================


class InstanceState(StrEnum):
    """EC2 instance state names."""

    RUNNING = "running"
    STOPPED = "stopped"
    PENDING = "pending"
    TERMINATED = "terminated"
    STOPPING = "stopping"
    SHUTTING_DOWN = "shutting-down"


# =============================================================================
# Filesystem Paths
# =============================================================================

SKYWARD_DIR: Final = "/opt/skyward"
VENV_DIR: Final = f"{SKYWARD_DIR}/.venv"


# =============================================================================
# Bootstrap Configuration
# =============================================================================

UV_INSTALL_URL: Final = "https://astral.sh/uv/install.sh"
DEFAULT_PYTHON: Final = "3.13"

# =============================================================================
# Serialization
# =============================================================================

COMPRESSED_MAGIC: Final = b"\x00CZ"
COMPRESSION_LEVEL: Final = 6
