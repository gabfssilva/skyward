"""Base classes for provider implementations.

Provides shared functionality across all cloud providers to reduce duplication.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from skyward.messages import ClusterId, InstanceId, InstanceMetadata
from skyward.spec import PoolSpec


# =============================================================================
# Base Cluster State
# =============================================================================


@dataclass
class BaseClusterState:
    """Base class for provider cluster state.

    Subclasses should add provider-specific fields.
    """

    cluster_id: ClusterId
    spec: PoolSpec

    instances: dict[InstanceId, InstanceMetadata] = field(default_factory=dict)
    pending_nodes: set[int] = field(default_factory=set)


# =============================================================================
# Exports
# =============================================================================

__all__ = ["BaseClusterState"]
