from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from casty import ActorRef, SpyEvent

if TYPE_CHECKING:
    from skyward.actors.session.messages import SessionMsg
    from skyward.actors.snapshot import PoolSnapshot


@dataclass(frozen=True, slots=True)
class LocalOutput:
    line: str
    stream: str = "stdout"


@dataclass(frozen=True, slots=True)
class _SetSession:
    ref: ActorRef[SessionMsg]


@dataclass(frozen=True, slots=True)
class _PollTick:
    pass


@dataclass(frozen=True, slots=True)
class _SnapshotReceived:
    snapshots: tuple[PoolSnapshot, ...]


type ConsoleInput = SpyEvent | LocalOutput | _SetSession | _PollTick | _SnapshotReceived
