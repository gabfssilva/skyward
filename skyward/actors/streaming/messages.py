from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StopMonitor:
    pass


type MonitorMsg = StopMonitor
