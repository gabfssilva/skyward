from __future__ import annotations

from dataclasses import dataclass, replace

from skyward.actors.messages import BoundsChanged, NodeId


@dataclass(frozen=True, slots=True)
class _State:
    desired: int
    current: frozenset[NodeId]
    pending: int = 0
    draining: int = 0
    consecutive_failures: int = 0
    last_requested: int = 0
    min_nodes: int = 1

    @property
    def effective(self) -> int:
        return len(self.current) + self.pending


def _apply_bounds(s: _State, msg: BoundsChanged) -> _State:
    return replace(s, min_nodes=msg.min, desired=msg.desired)
