"""RunPod cluster state tracking.

Manages runtime state for RunPod clusters including pod information.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from skyward.providers.base import BaseClusterState


# =============================================================================
# Cluster State
# =============================================================================


@dataclass
class RunPodClusterState(BaseClusterState):
    """Runtime state for a RunPod cluster.

    Tracks all information needed to manage a cluster's lifecycle
    including launched pods and spec.

    For multi-node deployments (nodes >= 2), uses Instant Clusters
    which provide high-speed networking (1600-3200 Gbps).
    """

    # RunPod-specific fields
    cloud_type: str = "secure"
    data_center_ids: tuple[str, ...] | str = "global"

    # SSH - RunPod pods use root by default
    username: str = "root"
    ssh_key_path: str = ""

    # Resolved GPU type
    gpu_type_id: str | None = None

    # Instant Cluster ID (when using Instant Clusters for multi-node)
    # None for single-node deployments using individual pods
    runpod_cluster_id: str | None = None

    # Pod tracking: node_id -> pod_id
    pod_ids: dict[int, str] = field(default_factory=dict)

    # Internal IPs for inter-node communication (Instant Clusters only)
    # node_id -> cluster_ip
    cluster_ips: dict[int, str] = field(default_factory=dict)

    # Pricing info (from pod creation)
    hourly_rate: float = 0.0
    on_demand_rate: float = 0.0

    # Hardware specs
    vcpus: int = 0
    memory_gb: float = 0.0
    gpu_count: int = 0
    gpu_model: str = ""
    gpu_vram_gb: int = 0

    region: str = ""

    @property
    def is_instant_cluster(self) -> bool:
        """True if this was created as an Instant Cluster (multi-node)."""
        return self.runpod_cluster_id is not None


# =============================================================================
# Exports
# =============================================================================

__all__ = ["RunPodClusterState"]
