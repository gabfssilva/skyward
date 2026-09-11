"""The one channel the machine talks back on.

Everything a node has to say that is not a task result is a line in
``events.jsonl``: the bootstrap writes through the shell helpers it leaves in
``emit.sh``, the worker writes through :class:`Journal`, and the daemon reads
both by tailing the one file. A file and not a socket, because a file survives
the link dropping — the machine keeps writing into it, and the reader comes back
to the line it had got to.

Written to be importable with nothing but ``msgspec``: it is the only module of
skyward that runs on both sides, and the node's side has no ``asyncssh``.
"""

from __future__ import annotations

import fcntl
import io
import os
import re
import threading
from contextvars import ContextVar
from typing import Literal

import msgspec
from msgspec import Struct

from skyward.worker.api import Stream, instance_info, policy

SKYWARD_DIR = "/opt/skyward"
EVENTS = f"{SKYWARD_DIR}/events.jsonl"
LOCK = f"{SKYWARD_DIR}/events.lock"

LINE_LIMIT = 64 * 1024
"""A line without a newline is written as it is once it reaches this many characters."""

ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

task: ContextVar[str | None] = ContextVar("task", default=None)
"""The task whose code is running right now, so its output can say so."""


class Phase(Struct, frozen=True, tag="phase", tag_field="type"):
    event: Literal["started", "completed", "failed"]
    phase: str
    error: str | None = None


class Console(Struct, frozen=True, tag="console", tag_field="type"):
    content: str
    task: str | None = None


class Metric(Struct, frozen=True, tag="metric", tag_field="type"):
    """One reading of one gauge, sampled on the machine and named for what it is.

    A gauge, not a log line: ``cpu`` at ``72.4`` replaces the last ``cpu``, it does
    not add to a history. The bootstrap's collectors write these on their own
    interval, independent of the worker, so a machine keeps saying how it is even
    while it is still coming up, or long after it has gone quiet.
    """

    name: str
    value: float


class Health(Struct, frozen=True, tag="health", tag_field="type"):
    reason: str


type NodeEvent = Phase | Console | Metric | Health

_decode = msgspec.json.Decoder(NodeEvent)


def parse(line: str) -> NodeEvent | None:
    """One line of the node's event log, or nothing if it is not one.

    A line can be half-written when the reader reaches it, and a container can be
    noisy on a channel nobody asked to be clean. Neither is worth failing over —
    the next line is along in a moment.
    """
    try:
        return _decode.decode(ANSI.sub("", line).encode())
    except msgspec.DecodeError:
        return None


def emit(event: NodeEvent) -> None:
    """Append one event, under the lock the bootstrap's shell helpers take.

    The bootstrap and the worker can be writing at the same moment — the last
    phase of one overlaps the first output of the other — and two appends without
    a lock interleave into a line that parses as neither.
    """
    line = msgspec.json.encode(event) + b"\n"
    with _writing:
        lock, journal = _opened()
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            view = memoryview(line)
            while view:
                view = view[os.write(journal, view) :]
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


_writing = threading.Lock()
_descriptors: tuple[str, str, int, int] | None = None


def _opened() -> tuple[int, int]:
    """The lock and the journal, opened once and reopened only when their paths change.

    ``O_APPEND`` keeps every write at the end even after the daemon truncates the file.
    """
    global _descriptors
    match _descriptors:
        case (lock_path, events_path, lock, journal) if (lock_path, events_path) == (LOCK, EVENTS):
            return lock, journal
        case (_, _, lock, journal):
            os.close(lock)
            os.close(journal)
        case None:
            pass
    lock = os.open(LOCK, os.O_WRONLY | os.O_CREAT, 0o644)
    journal = os.open(EVENTS, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    _descriptors = (LOCK, EVENTS, lock, journal)
    return lock, journal


class _Partial(threading.local):
    """The line one thread has started and not yet ended."""

    def __init__(self) -> None:
        self.chunks: list[str] = []
        self.size = 0


class Journal(io.TextIOBase):
    """The user's ``print``, as lines in the file the daemon is already reading.

    Output is buffered until a newline, because a line is the unit the log is made
    of: ``print(a, b)`` reaches here as four writes, and emitting each as its own
    event would scatter one line of the user's output across four.

    What the task's output policy silences is dropped here, where it was written,
    rather than shipped over SSH and thrown away at the other end.
    """

    def __init__(self, stream: Stream) -> None:
        self._stream: Stream = stream
        self._partial = _Partial()

    def write(self, s: str, /) -> int:
        partial = self._partial
        if "\n" not in s:
            partial.chunks.append(s)
            partial.size += len(s)
            if partial.size >= LINE_LIMIT:
                self.flush()
            return len(s)
        first, *rest = s.split("\n")
        *lines, last = rest
        self._emit("".join((*partial.chunks, first)))
        for line in lines:
            self._emit(line)
        partial.chunks = [last] if last else []
        partial.size = len(last)
        return len(s)

    def flush(self) -> None:
        partial = self._partial
        if partial.size:
            self._emit("".join(partial.chunks))
            partial.chunks = []
            partial.size = 0

    def _emit(self, content: str) -> None:
        if policy.get().allows(self._stream, instance_info()):
            emit(Console(content=content, task=task.get()))

    def isatty(self) -> bool:
        return False
