"""Direct interactive PTY into a session node over SSH (co-located, key-only).

The CLI opens its *own* asyncssh connection to the node and allocates a
PTY — the interactive byte stream never transits the server, the HTTP
control plane, or Casty. POSIX-only (Windows unsupported in Phase 1).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
import sys
from collections.abc import Awaitable, Callable
from typing import Any


def _term_size() -> tuple[int, int]:
    """Return the local terminal ``(columns, lines)`` (defaults 80x24)."""
    size = shutil.get_terminal_size((80, 24))
    return size.columns, size.lines


async def open_pty(
    host: str,
    port: int,
    user: str,
    key_path: str,
    password: str | None,
    command: str | None,
    env: dict[str, str] | None = None,
    *,
    connect_fn: Callable[[], Awaitable[Any]] | None = None,
) -> int:
    """Open a raw interactive PTY to a node over SSH.

    Pumps the local terminal to the remote process until it exits and
    returns the remote exit status. ``command=None`` runs the login shell.
    """
    import asyncssh

    connect = connect_fn or (
        lambda: asyncssh.connect(
            host, port=port, username=user,
            client_keys=[key_path] if key_path else [],
            password=password, known_hosts=None,
        )
    )
    conn = await connect()
    try:
        cols, rows = _term_size()
        proc = await conn.create_process(
            command,
            term_type=os.environ.get("TERM", "xterm-256color"),
            term_size=(cols, rows),
            encoding=None,
            env=env or {},
            stderr=asyncssh.STDOUT,
        )
        return await _interact(proc)
    finally:
        conn.close()
        with contextlib.suppress(Exception):
            await conn.wait_closed()


async def _interact(proc: Any) -> int:
    """Bridge the local terminal to a remote PTY process until it exits."""
    loop = asyncio.get_event_loop()
    is_tty = sys.stdin.isatty()
    stdin_fd = sys.stdin.fileno() if is_tty else -1
    old_attr = None

    if is_tty:
        import termios
        import tty

        old_attr = termios.tcgetattr(stdin_fd)
        tty.setraw(stdin_fd)

        def _read_stdin() -> None:
            try:
                data = os.read(stdin_fd, 4096)
            except OSError:
                data = b""
            if data:
                proc.stdin.write(data)
            else:
                with contextlib.suppress(Exception):
                    proc.stdin.write_eof()
                loop.remove_reader(stdin_fd)

        loop.add_reader(stdin_fd, _read_stdin)
        with contextlib.suppress(NotImplementedError, ValueError):
            loop.add_signal_handler(signal.SIGWINCH, lambda: _on_resize(proc))

    try:
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
        result = await proc.wait()
        return result.exit_status or 0
    finally:
        if is_tty:
            with contextlib.suppress(Exception):
                loop.remove_reader(stdin_fd)
            with contextlib.suppress(NotImplementedError, ValueError):
                loop.remove_signal_handler(signal.SIGWINCH)
            if old_attr is not None:
                import termios

                termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_attr)


def _on_resize(proc: Any) -> None:
    cols, rows = _term_size()
    proc.change_terminal_size(cols, rows)
