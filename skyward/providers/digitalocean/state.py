"""DigitalOcean cluster state.

Mutable state for tracking DigitalOcean resources during cluster lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass

from skyward.providers.base import BaseClusterState


# =============================================================================
# Cluster State
# =============================================================================


@dataclass
class DOClusterState(BaseClusterState):
    """Mutable state for a DigitalOcean cluster.

    Tracks all resources created during cluster lifecycle for proper cleanup.
    """

    # DigitalOcean-specific fields
    region: str = ""

    # SSH key info
    ssh_key_fingerprint: str = ""

    # Resolved instance configuration
    size_slug: str = ""
    os_image: str = ""
    username: str = "root"

    @property
    def droplet_ids(self) -> list[int]:
        """Get all droplet IDs as integers."""
        return [int(iid) for iid in self.instances]


# =============================================================================
# Exports
# =============================================================================

__all__ = ["DOClusterState"]
