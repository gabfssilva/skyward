from dataclasses import dataclass

from skyward.actors.messages import (
    BoundsChanged,
    DrainComplete,
    NodeBecameBusy,
    NodeBecameIdle,
    NodeJoined,
    PressureReport,
)


@dataclass(frozen=True, slots=True)
class _ScaleTick:
    pass


type AutoscalerMsg = (
    PressureReport
    | _ScaleTick
    | BoundsChanged
    | NodeBecameIdle
    | NodeBecameBusy
    | NodeJoined
    | DrainComplete
)
