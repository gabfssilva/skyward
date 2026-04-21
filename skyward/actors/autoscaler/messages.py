from dataclasses import dataclass

from skyward.actors.messages import (
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
    | NodeBecameIdle
    | NodeBecameBusy
    | NodeJoined
    | DrainComplete
)
