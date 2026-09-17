"""How an attempt that ran past its time is stopped, on the machine running it.

Nothing kills a thread, and a process pool one of whose children is killed breaks
for every task in flight in it — the attempts beside the one that ran too long would
be lost with it. So an attempt is stopped from the inside: :class:`Stop` is raised
where the function is running, the way ``KeyboardInterrupt`` is, and unwinds it like
any exception would, its ``finally`` blocks and context managers included.

In a thread it is raised through CPython's asynchronous exception; in a subprocess,
by a signal whose handler raises it. Either way it lands between two bytecodes, so a
function inside one long native call — a CUDA kernel, a blocking read in C — stops
when that call returns, and until then its slot stays taken.
"""

from __future__ import annotations

import ctypes
import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from types import FrameType

SIGNAL = signal.SIGUSR1
"""What a subprocess is sent to stop the attempt it is running."""

asked: set[str] = set()
"""Every attempt this worker was asked to stop, by execution — the ones not running yet included, which never start."""


class Stop(BaseException):
    """Raised inside an attempt the daemon asked to stop.

    A ``BaseException``, like ``KeyboardInterrupt``: a function's own ``except
    Exception`` must not swallow the one thing that is meant to end it.
    """


@contextmanager
def running(id: str) -> Iterator[None]:
    """Run the block as attempt ``id`` in this thread, where :func:`interrupt` can reach it.

    An attempt asked to stop before it got here does not start. On the way out, an
    interruption that arrived too late to land is taken back, so it cannot go off in
    whatever this thread runs next.
    """
    ident = threading.get_ident()
    with _lock:
        if id in asked:
            raise Stop
        _threads[id] = ident
    try:
        yield
    finally:
        with _lock:
            _threads.pop(id, None)
            _raise_in(ident, None)


def interrupt(id: str) -> bool:
    """Raise :class:`Stop` in the thread running ``id``, once. ``False`` if no thread is running it."""
    with _lock:
        ident = _threads.pop(id, None)
        if ident is None:
            return False
        _raise_in(ident, Stop)
        return True


def arm() -> None:
    """Make this subprocess stoppable. Called once per child, by the pool's initializer."""
    signal.signal(SIGNAL, _signalled)


@contextmanager
def current(id: str) -> Iterator[None]:
    """Run the block as attempt ``id`` in this subprocess: a stop signal that arrives meanwhile is for it."""
    global _current, _owed
    _current = id
    try:
        yield
    finally:
        _current = None
        _owed = False


@contextmanager
def deferred() -> Iterator[None]:
    """Hold a stop back for the length of the block, and raise it once the block is over.

    The block is an exchange on the pipe to the worker. A stop raised in the middle of
    one would leave its reply unread, and the next exchange would read that reply as
    its own.
    """
    global _holding, _owed
    _holding += 1
    try:
        yield
    finally:
        _holding -= 1
        if not _holding and _owed and _current is not None:
            _owed = False
            raise Stop


_threads: dict[str, int] = {}
_lock = threading.Lock()
_current: str | None = None
_holding = 0
_owed = False


def _raise_in(thread: int, exception: type[BaseException] | None) -> None:
    """Set, or with ``None`` clear, the exception a thread raises at its next bytecode."""
    ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread), None if exception is None else ctypes.py_object(exception))


def _signalled(signum: int, frame: FrameType | None) -> None:
    global _owed
    if _current is None:
        return
    if _holding:
        _owed = True
        return
    raise Stop
