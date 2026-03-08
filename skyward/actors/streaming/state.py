from __future__ import annotations

from dataclasses import dataclass

from casty import ActorRef

from skyward.actors.messages import BootstrapDone, NodeInstance


@dataclass(frozen=True, slots=True)
class MonitorState:
    info: NodeInstance
    transport: ActorRef
    event_listener: ActorRef
    reply_to: ActorRef[BootstrapDone]
    bootstrap_signaled: bool = False
