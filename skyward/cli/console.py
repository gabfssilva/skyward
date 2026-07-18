"""``sky console`` and ``sky repl`` — a terminal on one of a compute's machines.

The SSH connections belong to the daemon, so this side never opens one. It mints
a session id, opens the same pair of half-duplex streams a forwarded port uses —
keystrokes up, whatever the terminal paints down — and spends the rest of the
session copying bytes between them and this process's own terminal.

Which means the local terminal has to stop being clever. In its usual mode the
tty buffers a line, echoes it, and eats Ctrl-C to signal *this* process; all
three belong to the shell at the far end, so the tty is put in raw mode for the
session and put back, whatever happens, when it ends.
"""

from __future__ import annotations

import asyncio
import os
import sys
import termios
import tty
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from skyward.cli import app
from skyward.cli._client import call
from skyward.runtime.bootstrap import PYTHON
from skyward.sdk.client import Client

CHUNK = 65536


@app.command(name="console")
def console(
    ref: Annotated[str, Parameter(help="A compute id or name")],
    *,
    node: Annotated[str | None, Parameter(help="The node to open the terminal on; omit for the first ready one")] = None,
    command: Annotated[str | None, Parameter(help="What to run; omit for the login shell")] = None,
    url: Annotated[str | None, Parameter(help="Daemon URL")] = None,
    database: Annotated[Path | None, Parameter(help="Embedded daemon database")] = None,
) -> None:
    """Open an interactive shell on one of a compute's machines.

    The same machine for the whole session, and the same one again if you name it:
    a terminal is somebody sitting at a computer, not a load-balanced request.
    """
    call(lambda client: _attach(client, ref, node, command), url=url, database=database)


@app.command(name="repl")
def repl(
    ref: Annotated[str, Parameter(help="A compute id or name")],
    *,
    node: Annotated[str | None, Parameter(help="The node to open the REPL on; omit for the first ready one")] = None,
    url: Annotated[str | None, Parameter(help="Daemon URL")] = None,
    database: Annotated[Path | None, Parameter(help="Embedded daemon database")] = None,
) -> None:
    """Open a Python REPL on one of a compute's machines.

    ``console`` with the interpreter the node was bootstrapped with, which is the
    one holding the compute's dependencies — not whatever ``python`` resolves to on
    a machine that was never meant to be logged into.
    """
    console(ref, node=node, command=PYTHON, url=url, database=database)


@contextmanager
def raw() -> Iterator[None]:
    """Hand this terminal over for the length of a session, and take it back after.

    A pipe is left alone: there is no tty to make raw, and a ``sky console`` on the
    end of one is a script, which wants its bytes untouched anyway.
    """
    if not sys.stdin.isatty():
        yield
        return

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def size() -> tuple[int, int]:
    with suppress(OSError):
        window = os.get_terminal_size()
        return window.columns, window.lines
    return 80, 24


async def _keystrokes() -> AsyncIterator[bytes]:
    """This process's stdin, as it is typed.

    Read through the event loop rather than a thread so that the pump ends when the
    session does: a blocking read would still be sitting on the keyboard long after
    the shell exited, holding the command open until somebody pressed a key.
    """
    reader = asyncio.StreamReader()
    loop = asyncio.get_running_loop()
    transport, _ = await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    try:
        while data := await reader.read(CHUNK):
            yield data
    finally:
        transport.close()


async def _attach(client: Client, ref: str, node: str | None, command: str | None) -> None:
    cid = uuid.uuid4().hex
    with raw():
        pump = asyncio.create_task(client.shell_up(ref, cid, node, command, os.environ.get("TERM", "xterm-256color"), size(), _keystrokes()))
        try:
            async for chunk in client.shell_down(ref, cid):
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
        finally:
            pump.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await pump


__all__ = ["console", "raw", "repl", "size"]
