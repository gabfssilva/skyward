"""Centralized constants and enums for Skyward.

All magic strings, paths, and configuration constants are defined here
to ensure consistency and enable type-safe usage throughout the codebase.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

# =============================================================================
# AWS Resource Tags
# =============================================================================


class SkywardTag(StrEnum):
    """AWS resource tag keys used by Skyward."""

    MANAGED = "skyward:managed"
    REQUIREMENTS_HASH = "skyward:requirements-hash"
    CLUSTER_ID = "skyward:cluster-id"
    NODE_INDEX = "skyward:node-index"


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
SYSTEMD_SERVICE: Final = "/etc/systemd/system/skyward-rpyc.service"


# =============================================================================
# Bootstrap Configuration
# =============================================================================

SSM_WAIT_SECONDS: Final = 5
UV_INSTALL_URL: Final = "https://astral.sh/uv/install.sh"
DEFAULT_PYTHON: Final = "3.13"
RPYC_PORT: Final = 18861

# Timeouts (in seconds)
SSM_AGENT_TIMEOUT: Final = 600
INSTANCE_RUNNING_WAIT_DELAY: Final = 2
INSTANCE_RUNNING_MAX_ATTEMPTS: Final = 300


# =============================================================================
# Default Resource Names
# =============================================================================

DEFAULT_INSTANCE_NAME: Final = "skyward-worker"


# =============================================================================
# Serialization
# =============================================================================

COMPRESSED_MAGIC: Final = b"\x00CZ"
COMPRESSION_LEVEL: Final = 6
