"""Base classes for provider implementations.

Provides shared functionality across all cloud providers to reduce duplication.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from skyward.actors.messages import ClusterId, InstanceId, InstanceMetadata
from skyward.api.spec import PoolSpec

@dataclass
class BaseClusterState:
    """Base class for provider cluster state.

    Subclasses should add provider-specific fields.
    """

    cluster_id: ClusterId
    spec: PoolSpec

    instances: dict[InstanceId, InstanceMetadata] = field(default_factory=dict)
    pending_nodes: set[int] = field(default_factory=set)

    def add_instance(self, info: InstanceMetadata) -> None:
        self.instances[info.id] = info
        self.pending_nodes.discard(info.node)

    @property
    def instance_ids(self) -> list[InstanceId]:
        return list(self.instances.keys())
