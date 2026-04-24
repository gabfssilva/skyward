from __future__ import annotations

from dataclasses import dataclass

from skyward.actors.messages import (
    BoundsChanged,
    DesiredCountChanged,
    DrainComplete,
    NodeJoined,
    ReapIdleNodes,
    ReconcilerNodeLost,
    ScaleDownComplete,
    ScaleUpComplete,
    ScaleUpFailed,
)


@dataclass(frozen=True, slots=True)
class _ReconcileTick:
    pass


type ReconcilerMsg = (
    DesiredCountChanged
    | BoundsChanged
    | ReconcilerNodeLost
    | NodeJoined
    | DrainComplete
    | ScaleUpComplete
    | ScaleUpFailed
    | ScaleDownComplete
    | ReapIdleNodes
    | _ReconcileTick
)
