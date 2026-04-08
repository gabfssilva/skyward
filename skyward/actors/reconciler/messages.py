from __future__ import annotations

from dataclasses import dataclass

from skyward.actors.messages import (
    DesiredCountChanged,
    DrainComplete,
    NodeJoined,
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
    | ReconcilerNodeLost
    | NodeJoined
    | DrainComplete
    | ScaleUpComplete
    | ScaleUpFailed
    | ScaleDownComplete
    | _ReconcileTick
)
