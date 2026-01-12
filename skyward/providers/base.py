"""Base classes for provider implementations.

Provides shared functionality across all cloud providers to reduce duplication.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from skyward.events import ClusterId, InstanceId, InstanceInfo
from skyward.spec import PoolSpec


# =============================================================================
# Base Cluster State
# =============================================================================


@dataclass
class BaseClusterState:
    """Base class for provider cluster state.

    Provides common instance tracking functionality shared by all providers.
    Subclasses should add provider-specific fields.
    """

    cluster_id: ClusterId
    spec: PoolSpec

    # Instance tracking
    instances: dict[InstanceId, InstanceInfo] = field(default_factory=dict)
    pending_nodes: set[int] = field(default_factory=set)

    def add_instance(self, info: InstanceInfo) -> None:
        """Register a new instance."""
        self.instances[info.id] = info
        self.pending_nodes.discard(info.node)

    def remove_instance(self, instance_id: InstanceId) -> InstanceInfo | None:
        """Remove an instance, returns the removed info."""
        return self.instances.pop(instance_id, None)

    def get_instance_for_node(self, node_id: int) -> InstanceInfo | None:
        """Get instance info for a specific node."""
        for info in self.instances.values():
            if info.node == node_id:
                return info
        return None

    @property
    def instance_ids(self) -> list[str]:
        """List of all instance IDs."""
        return list(self.instances.keys())

    @property
    def is_complete(self) -> bool:
        """Whether all nodes have instances."""
        return len(self.instances) >= self.spec.nodes and not self.pending_nodes


# =============================================================================
# Exports
# =============================================================================

__all__ = ["BaseClusterState"]
