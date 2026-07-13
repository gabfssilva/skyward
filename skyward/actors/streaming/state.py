from __future__ import annotations

from dataclasses import dataclass

from casty import ActorRef

from skyward.actors.messages import BootstrapDone, NodeInstance
from skyward.api import events


def _no_emit(_event: events.SessionEvent) -> None:
    pass


@dataclass(frozen=True, slots=True)
class MonitorState:
    info: NodeInstance
    transport: ActorRef
    event_listener: ActorRef
    reply_to: ActorRef[BootstrapDone]
    bootstrap_signaled: bool = False
    emit: events.Emit = _no_emit
    pool_name: str = ""
