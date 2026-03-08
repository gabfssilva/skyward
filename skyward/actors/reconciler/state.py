from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skyward.api.model import Cluster

from skyward.actors.messages import NodeId


@dataclass(frozen=True, slots=True)
class _State:
    desired: int
    current: frozenset[NodeId]
    pending: frozenset[NodeId]
    draining: frozenset[NodeId]
    next_node_id: int
    instance_map: dict[NodeId, str]
    cluster: Cluster
