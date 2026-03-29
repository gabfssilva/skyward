from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skyward.api.events import Log, SessionEvent
    from skyward.api.views import SessionView


@dataclass(frozen=True, slots=True)
class LocalOutput:
    line: str
    stream: str = "stdout"


@dataclass(frozen=True, slots=True)
class ViewUpdated:
    view: SessionView


@dataclass(frozen=True, slots=True)
class EventReceived:
    event: SessionEvent


@dataclass(frozen=True, slots=True)
class LogReceived:
    log: Log.Emitted


type ConsoleInput = ViewUpdated | EventReceived | LogReceived | LocalOutput
