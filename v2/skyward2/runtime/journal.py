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
import re
from contextvars import ContextVar
from typing import Literal

import msgspec
from msgspec import Struct

SKYWARD_DIR = "/opt/skyward"
EVENTS = f"{SKYWARD_DIR}/events.jsonl"
LOCK = f"{SKYWARD_DIR}/events.lock"

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


type NodeEvent = Phase | Console

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
    with open(LOCK, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with open(EVENTS, "ab") as journal:
            journal.write(line)


class Journal(io.TextIOBase):
    """The user's ``print``, as lines in the file the daemon is already reading.

    Output is buffered until a newline, because a line is the unit the log is made
    of: ``print(a, b)`` reaches here as four writes, and emitting each as its own
    event would scatter one line of the user's output across four.
    """

    def __init__(self) -> None:
        self._partial = ""

    def write(self, s: str, /) -> int:
        *lines, self._partial = (self._partial + s).split("\n")
        for line in lines:
            emit(Console(content=line, task=task.get()))
        return len(s)

    def flush(self) -> None:
        if self._partial:
            emit(Console(content=self._partial, task=task.get()))
            self._partial = ""

    def isatty(self) -> bool:
        return False
