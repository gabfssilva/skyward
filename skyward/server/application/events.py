from __future__ import annotations

import contextlib
import shlex
from collections.abc import AsyncGenerator

from skyward.server.application.ssh import Ssh
from skyward.worker.journal import EVENTS, LOCK, NodeEvent, parse

ROTATE_BYTES = 16 * 1024 * 1024
"""How much of a node's log the daemon reads before it tries to empty the file."""

ROTATE_SECONDS = 30.0
"""The least time between two tries, so a log that keeps growing is not checked on every line."""


async def truncate(ssh: Ssh, offset: int, sudo: str) -> bool:
    """Empty the machine's log, if it holds exactly the ``offset`` bytes already read.

    Nothing is appended between the size check and the truncation, because every
    writer takes the same lock to append. A file that grew past ``offset`` holds
    lines not yet read, and is left alone for a later try.

    Returns
    -------
    bool
        Whether the file was emptied, and the tail should start again from byte 0.
    """
    script = f'[ "$(stat -c %s {EVENTS})" = {offset} ] && : > {EVENTS}'
    result = await ssh.run(f"{sudo}flock {LOCK} sh -c {shlex.quote(script)}")
    return result.exit_code == 0


async def events(ssh: Ssh, offset: int = 0) -> AsyncGenerator[tuple[int, NodeEvent]]:
    """Everything the machine has to say, from byte ``offset`` of its log on.

    The tail follows the file rather than the process, so it survives the
    bootstrap ending, the worker starting, and the two of them writing at once.
    A dropped link ends the iteration; the caller comes back with the offset it
    got to, and nothing that happened in between is lost, because it is on disk.

    An offset in bytes and not a line number, because ``tail`` seeks straight to a
    byte and has to read every line before a line number to find it: the whole log,
    again, on every reconnect, and some providers drop a link every few minutes.
    Only a whole line moves the offset — one the drop cut in half is read again,
    whole, when the caller comes back.

    A link that has given up is not a drop, and is not swallowed here: a channel
    that raises :class:`SshUnavailableError` will never carry another line, and a
    caller that came back for the next one would be coming back forever.

    Yields
    ------
    tuple[int, NodeEvent]
        The offset just past the line, and what was on it. The offset is what
        makes the resume exact.
    """
    with contextlib.suppress(ConnectionError):
        async for line in ssh.stream(f"stdbuf -oL tail -s 0.1 -c +{offset + 1} -F {EVENTS} 2>/dev/null"):
            if not line.endswith(b"\n"):
                return
            offset += len(line)
            if event := parse(line[:-1].decode(errors="replace")):
                yield offset, event
