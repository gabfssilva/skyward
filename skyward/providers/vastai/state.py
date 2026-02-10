"""Vast.ai cluster state tracking.

Manages runtime state for Vast.ai clusters including overlay networks
and instance information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType

from skyward.messages import InstanceMetadata
from skyward.spec import PoolSpec


# =============================================================================
# Cluster State
# =============================================================================


@dataclass(frozen=True, slots=True)
class InstancePricing:
    """Pricing info for a launched instance."""

    hourly_rate: float
    on_demand_rate: float
    gpu_name: str
    gpu_count: int


@dataclass(frozen=True, slots=True)
class VastAIClusterState:
    """Runtime state for a Vast.ai cluster.

    Tracks all information needed to manage a cluster's lifecycle
    including overlay networks, launched instances, and spec.
    """

    cluster_id: str
    spec: PoolSpec

    # SSH key info
    ssh_key_id: int | None = None
    ssh_public_key: str = ""

    # Overlay network state
    overlay_name: str | None = None
    overlay_cluster_id: int | None = None

    # Docker image
    docker_image: str | None = None

    # Geolocation
    geolocation: str | None = None

    # Instances and pricing
    instances: MappingProxyType[str, InstanceMetadata] = field(default_factory=lambda: MappingProxyType({}))
    pending_nodes: frozenset[int] = frozenset()
    instance_pricing: MappingProxyType[str, InstancePricing] = field(default_factory=lambda: MappingProxyType({}))


# =============================================================================
# Exports
# =============================================================================

__all__ = ["VastAIClusterState", "InstancePricing"]
