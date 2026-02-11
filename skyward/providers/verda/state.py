"""Verda cluster state tracking.

Manages runtime state for Verda clusters including instance information.
"""

from __future__ import annotations

from dataclasses import dataclass

from skyward.providers.base import BaseClusterState


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
    ssh_key_path: str = ""

    # Startup script ID (for cleanup)
    startup_script_id: str | None = None

    # Resolved instance type and image
    instance_type: str | None = None
    os_image: str | None = None

    # Username for SSH
    username: str = "root"

    # Pricing info (from instance type resolution)
    hourly_rate: float = 0.0  # Actual rate (spot or on-demand)
    on_demand_rate: float = 0.0  # On-demand rate for savings calc

    # Hardware specs (from instance type resolution)
    vcpus: int = 0
    memory_gb: float = 0.0
    gpu_count: int = 0
    gpu_model: str = ""
    gpu_vram_gb: int = 0


