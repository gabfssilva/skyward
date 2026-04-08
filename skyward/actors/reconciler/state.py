from __future__ import annotations

from dataclasses import dataclass

from skyward.actors.messages import NodeId


@dataclass(frozen=True, slots=True)
class _State:
    desired: int
    current: frozenset[NodeId]
    pending: int = 0
    draining: int = 0
    consecutive_failures: int = 0

    @property
    def effective(self) -> int:
        return len(self.current) + self.pending
