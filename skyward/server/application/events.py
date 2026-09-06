from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator

from skyward.server.application.ssh import Ssh
from skyward.worker.journal import EVENTS, NodeEvent, parse


async def events(ssh: Ssh, first: int = 1) -> AsyncIterator[tuple[int, NodeEvent]]:
    """Everything the machine has to say, from line ``first`` on.

    The tail follows the file rather than the process, so it survives the
    bootstrap ending, the worker starting, and the two of them writing at once.
    A dropped link ends the iteration; the caller comes back with the line it got
    to, and nothing that happened in between is lost, because it is on disk.

    A link that has given up is not a drop, and is not swallowed here: a channel
    that raises :class:`SshUnavailableError` will never carry another line, and a
    caller that came back for the next one would be coming back forever.

    Yields
    ------
    tuple[int, NodeEvent]
        The line number and what was on it. The number is what makes the resume
        exact.
    """
    line = first - 1
    with contextlib.suppress(ConnectionError):
        async for raw in ssh.stream(f"stdbuf -oL tail -s 0.1 -n +{first} -F {EVENTS} 2>/dev/null"):
            line += 1
            if event := parse(raw):
                yield line, event
