from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator

from skyward.server.application.ssh import Ssh
from skyward.worker.journal import EVENTS, NodeEvent, parse


async def events(ssh: Ssh, offset: int = 0) -> AsyncIterator[tuple[int, NodeEvent]]:
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
