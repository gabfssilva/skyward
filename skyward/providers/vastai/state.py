"""Vast.ai cluster state tracking.

Manages runtime state for Vast.ai clusters including overlay networks
and instance information.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from skyward.providers.base import BaseClusterState


# =============================================================================
# Cluster State
# =============================================================================


@dataclass
class InstancePricing:
    """Pricing info for a launched instance."""

    hourly_rate: float  # Actual rate (spot or on-demand)
    on_demand_rate: float  # On-demand rate for savings calculation
    gpu_name: str
    gpu_count: int


@dataclass
class VastAIClusterState(BaseClusterState):
    """Runtime state for a Vast.ai cluster.

    Tracks all information needed to manage a cluster's lifecycle
    including overlay networks, launched instances, and spec.
    """

    # VastAI-specific fields
    geolocation: str | None = None

    # SSH key info
    ssh_key_id: int | None = None
    ssh_public_key: str = ""

    # Overlay network state
    overlay_name: str | None = None
    overlay_cluster_id: int | None = None
    overlay_ips: dict[int, str] = field(default_factory=dict)  # instance_id -> overlay IP
    overlay_ifaces: dict[int, str] = field(default_factory=dict)  # instance_id -> interface

    # Docker image
    docker_image: str | None = None

    # Pricing info per instance (instance_id -> pricing)
    instance_pricing: dict[str, InstancePricing] = field(default_factory=dict)


# =============================================================================
# Exports
# =============================================================================

__all__ = ["VastAIClusterState", "InstancePricing"]
