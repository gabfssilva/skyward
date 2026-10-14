"""What went wrong lately, and which of it the reader has already seen.

A notification is a failure the daemon already wrote in its event log — a compute
that degraded or could not be deleted, a machine lost or failed, a task that
failed or ran out of time — so the daemon keeps nothing new for it. The log is
read for the last :data:`WINDOW`, the newest :data:`SAMPLE` failures at most, and
folded into one notification per compute, kind and reason: a rank lost five times
for the same reason is one line that says so five times over, not five lines.

What has been seen belongs to the reader. :class:`Seen` keeps the sequences
dismissed in a file of the reader's, per daemon, each with when it was dismissed,
and forgets the ones dismissed more than a :data:`WINDOW` ago: what was seen then
failed earlier still, and no read returns it any more.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import groupby
from pathlib import Path

import msgspec

from skyward.api.v1 import (
    ComputeDegradedEvent,
    ComputeDeletionFailedEvent,
    LogEntryResource,
    NodeStateEvent,
    Page,
    TaskStateEvent,
)
from skyward.core.client import Client

FAILURES: tuple[str, ...] = ("compute.degraded", "compute.deletion_failed", "node.lost", "node.failed", "task.failed", "task.timed_out")
"""The frame names of the events that are a failure."""

WINDOW = timedelta(hours=24)
SAMPLE = 200
SEEN = Path.home() / ".skyward" / "seen.json"


@dataclass(frozen=True, slots=True)
class Notification:
    """The failures of one kind on one compute for one reason, the newest first."""

    compute: str
    kind: str
    """The frame name: ``node.lost``, ``task.failed``, ``compute.degraded``…"""
    reason: str | None
    sequences: tuple[int, ...]
    at: datetime
    """When the newest of them was recorded."""


@dataclass(frozen=True, slots=True)
class Seen:
    """The failures one reader dismissed on one daemon, kept in a file of the reader's."""

    path: Path
    daemon: str

    def read(self) -> frozenset[int]:
        return frozenset(self._every().get(self.daemon, {}))

    def mark(self, sequences: Iterable[int], now: datetime) -> None:
        """Note ``sequences`` as seen at ``now``, and forget what was seen more than a :data:`WINDOW` before it.

        The file is read again rather than trusted from a previous read, so two
        readers of one daemon add to each other's dismissals instead of replacing
        them, and it is replaced whole, so a reader never finds half of one.
        """
        every = self._every()
        kept = {sequence: at for sequence, at in every.get(self.daemon, {}).items() if at >= now - WINDOW} | dict.fromkeys(sequences, now)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        written = self.path.with_suffix(f".{os.getpid()}.tmp")
        written.write_bytes(msgspec.json.encode({**every, self.daemon: kept}))
        written.replace(self.path)

    def _every(self) -> dict[str, dict[int, datetime]]:
        if not self.path.exists():
            return {}
        return msgspec.json.decode(self.path.read_bytes(), type=dict[str, dict[int, datetime]])


async def window(client: Client, now: datetime, compute: str | None = None) -> tuple[LogEntryResource, ...]:
    """The failures recorded in the last :data:`WINDOW`, on one compute or on all of them, the newest first."""
    page = await client.call("GET", "/v1/events/log", Page[LogEntryResource], types=FAILURES, limit=SAMPLE, compute=compute)
    return tuple(entry for entry in page.items if entry.at >= now - WINDOW)


def unseen(entries: Iterable[LogEntryResource], seen: Set[int]) -> tuple[Notification, ...]:
    """The failures not yet seen, folded by compute, kind and reason, the group with the newest first."""

    def failure(entry: LogEntryResource) -> _Failure | None:
        match entry.data:
            case ComputeDegradedEvent(compute=compute, error=error) | ComputeDeletionFailedEvent(compute=compute, error=error):
                return _Failure(compute, entry.type, error, entry.sequence, entry.at)
            case NodeStateEvent(compute=compute, error=error) | TaskStateEvent(compute=compute, error=error):
                return _Failure(compute, entry.type, error, entry.sequence, entry.at)
            case _:
                return None

    def notification(group: tuple[_Failure, ...]) -> Notification:
        newest = group[0]
        return Notification(newest.compute, newest.kind, newest.reason, tuple(failure.sequence for failure in group), newest.at)

    failures = sorted((found for entry in entries if entry.sequence not in seen and (found := failure(entry)) is not None), key=_Failure.key)
    folded = (notification(tuple(group)) for _, group in groupby(failures, key=_Failure.key))
    return tuple(sorted(folded, key=lambda notification: notification.at, reverse=True))


def describe(kind: str) -> str:
    """A failure's kind as a person says it: ``node lost``, ``task timed out``, ``degraded``."""
    return kind.removeprefix("compute.").replace(".", " ").replace("_", " ")


@dataclass(frozen=True, slots=True)
class _Failure:
    compute: str
    kind: str
    reason: str | None
    sequence: int
    at: datetime

    def key(self) -> tuple[str, str, str]:
        return self.compute, self.kind, self.reason or ""


__all__ = ["FAILURES", "SAMPLE", "SEEN", "WINDOW", "Notification", "Seen", "describe", "unseen", "window"]

