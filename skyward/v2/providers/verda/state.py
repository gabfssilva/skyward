"""Verda cluster state tracking.

Manages runtime state for Verda clusters including instance information.
"""

from __future__ import annotations

from dataclasses import dataclass

from skyward.v2.providers.base import BaseClusterState


# =============================================================================
# Cluster State
# =============================================================================


@dataclass
class VerdaClusterState(BaseClusterState):
    """Runtime state for a Verda cluster.

    Tracks all information needed to manage a cluster's lifecycle
    including launched instances, startup scripts, and spec.
    """

    # Verda-specific fields
    region: str = ""

    # SSH key info
    ssh_key_id: str | None = None

    # Startup script ID (for cleanup)
    startup_script_id: str | None = None

    # Resolved instance type and image
    instance_type: str | None = None
    os_image: str | None = None

    # Username for SSH
    username: str = "root"


# =============================================================================
# Exports
# =============================================================================

__all__ = ["VerdaClusterState"]
