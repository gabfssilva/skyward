from dataclasses import dataclass

from skyward.actors.messages import PressureReport


@dataclass(frozen=True, slots=True)
class _ScaleTick:
    pass


type AutoscalerMsg = PressureReport | _ScaleTick
