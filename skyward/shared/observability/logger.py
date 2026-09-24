"""A loguru-shaped logger over the standard library, and nothing else.

The shape is loguru's — ``bind`` returns a child carrying extra fields, ``add``
attaches a sink and hands back an id, ``remove`` takes it away — but the machinery
underneath is ``logging``::

    from skyward.shared.observability import logger

    log = logger.bind(component="pool")
    log.info("started {n} nodes", n=4)

Every record goes to one non-propagating logger of its own, so adding a sink here
cannot disturb whatever logging the host application has already set up, and
nothing the daemon logs leaks into it. Bound fields ride on the record
as ``extras``, out of reach of the reserved
``LogRecord`` attributes, and a patcher may fold them into the formatted line.

A record is not written on the thread that logs it: the logger holds a queue, and
one background thread hands what arrives to the sinks, so a slow disk or a
rollover never stalls the event loop that logged. With no sink attached the queue
is detached too, and a record falls through to ``logging.lastResort``. A record is
settled before it is queued: its message formatted, and the exception it carries,
if any, turned into a :class:`~skyward.shared.observability.logfile.Failure` and
the traceback text a console prints.

A library's logger can be made to speak through this one with ``capture``, from a
level up: its records reach the same sinks, under the library's own name.
"""

from __future__ import annotations

import atexit
import copy
import logging
import logging.handlers
import queue
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TextIO

from skyward.shared.observability.logfile import failure

TRACE = 5
logging.addLevelName(TRACE, "TRACE")

NAME = "skyward.log"
"""The logger every record here goes to.

Deliberately not ``skyward``: a non-propagating logger at ``skyward`` would be an
ancestor of every ``skyward.*`` logger the host application or a library happens to
own, and would swallow their records on the way to whatever they configured. This
one is nobody's ancestor.
"""

_root = logging.getLogger(NAME)

type Patcher = Callable[[logging.LogRecord], None]

_FORMAT = "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(module)s:%(funcName)s:%(lineno)d%(_ctx)s - %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _exception(value: object) -> BaseException | bool:
    match value:
        case BaseException() as exc:
            return exc
        case _:
            return bool(value)


def _format(message: str, args: tuple[object, ...], kwargs: Mapping[str, object]) -> str:
    match (args, kwargs):
        case (_, {**fields}) if fields:
            return message.format(**fields)
        case (positional, _) if positional:
            return message.format(*positional)
        case _:
            return message


class Logger:
    """A logger, optionally carrying bound fields. Immutable — ``bind`` returns a child."""

    __slots__ = ("_fields",)

    def __init__(self, fields: Mapping[str, object] = MappingProxyType({})) -> None:
        self._fields = fields

    def bind(self, **fields: object) -> Logger:
        """Return a child logger carrying this logger's fields plus ``fields``."""
        return Logger(MappingProxyType({**self._fields, **fields}))

    @property
    def fields(self) -> Mapping[str, object]:
        """The bound fields every record from this logger carries."""
        return self._fields

    def _log(self, level: int, message: str, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        exc_info = _exception(kwargs.pop("exc_info", False))
        if not _root.isEnabledFor(level):
            return
        _root.log(
            level,
            _format(message, args, kwargs),
            exc_info=exc_info,
            stacklevel=3,
            extra={"extras": self._fields, "_ctx": ""},
        )

    def trace(self, message: str, /, *args: object, **kwargs: object) -> None:
        """Log at ``TRACE``."""
        self._log(TRACE, message, args, kwargs)

    def debug(self, message: str, /, *args: object, **kwargs: object) -> None:
        """Log at ``DEBUG``."""
        self._log(logging.DEBUG, message, args, kwargs)

    def info(self, message: str, /, *args: object, **kwargs: object) -> None:
        """Log at ``INFO``."""
        self._log(logging.INFO, message, args, kwargs)

    def warning(self, message: str, /, *args: object, **kwargs: object) -> None:
        """Log at ``WARNING``."""
        self._log(logging.WARNING, message, args, kwargs)

    def error(self, message: str, /, *args: object, **kwargs: object) -> None:
        """Log at ``ERROR``."""
        self._log(logging.ERROR, message, args, kwargs)

    def exception(self, message: str, /, *args: object, **kwargs: object) -> None:
        """Log at ``ERROR`` with the active exception attached."""
        self._log(logging.ERROR, message, args, {**kwargs, "exc_info": True})

    def add(self, sink: TextIO | logging.Handler, *, level: str = "DEBUG", filter: str | None = None) -> int:
        """Attach a sink — a stream, or a handler such as a log file — and return its id.

        Parameters
        ----------
        sink
            An open text stream, written as formatted lines, or a handler that writes
            records its own way.
        level
            Minimum severity the sink accepts.
        filter
            Logger-name prefix the record must match.
        """
        global _counter

        numeric = logging.getLevelNamesMapping().get(level.upper(), logging.DEBUG)
        match sink:
            case logging.Handler() as handler:
                handler.setLevel(numeric)
            case stream:
                handler = _stream_handler(stream, level=numeric)

        if filter:
            handler.addFilter(logging.Filter(filter))

        _counter += 1
        _handlers[_counter] = handler
        _rewire()
        return _counter

    def remove(self, handler_id: int | None = None) -> None:
        """Detach the sink with this id, or every sink when given none."""
        if handler_id is None:
            _handlers.clear()
        else:
            _handlers.pop(handler_id, None)
        _rewire()

    def capture(self, name: str, level: str = "WARNING") -> None:
        """Have the library logger ``name`` speak through this one, from ``level`` up."""
        target = logging.getLogger(name)
        if not any(isinstance(handler, _Bridge) for handler in target.handlers):
            target.addHandler(_Bridge(logging.getLevelNamesMapping().get(level.upper(), logging.WARNING)))

    def release(self, name: str) -> None:
        """Undo ``capture``: the library logger goes back to whatever it did before."""
        target = logging.getLogger(name)
        for handler in [handler for handler in target.handlers if isinstance(handler, _Bridge)]:
            target.removeHandler(handler)

    def enable(self, name: str = NAME) -> None:
        """Re-enable a logger silenced by ``disable``."""
        target = logging.getLogger(name)
        target.disabled = False
        target.setLevel(TRACE)

    def disable(self, name: str = NAME) -> None:
        """Silence a logger and everything under it."""
        logging.getLogger(name).disabled = True

    def configure(self, *, patcher: Patcher | None = None) -> None:
        """Install a callback that rewrites every record before it reaches a sink."""
        global _patcher
        _patcher = patcher


class _Front(logging.handlers.QueueHandler):
    """The queue's end on the logging thread, which settles a record before handing it over.

    ``QueueHandler`` would fold the traceback into the message and drop the exception;
    this keeps the message as it was said and the exception as a value, and leaves
    the traceback text where a formatter prints it.
    """

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        settled = copy.copy(record)
        settled.message = settled.msg = record.getMessage()
        settled.args = None
        settled.exc_info = None
        settled.exc_text = None
        settled.stack_info = None
        if record.exc_info and (exc := record.exc_info[1]) is not None:
            failed = failure(exc)
            settled.__dict__["failure"] = failed
            settled.exc_text = failed.traceback
        return settled


class _Bridge(logging.Handler):
    """Hands a library's record to this logger, whose filters and sinks take it from there."""

    def emit(self, record: logging.LogRecord) -> None:
        _root.handle(record)


_counter = 0
_handlers: dict[int, logging.Handler] = {}
_patcher: Patcher | None = None
_queue: queue.SimpleQueue[logging.LogRecord] = queue.SimpleQueue()
_front = _Front(_queue)
_listener: logging.handlers.QueueListener | None = None


def _rewire() -> None:
    global _listener
    _stop()
    if not _handlers:
        _root.removeHandler(_front)
        return
    _listener = logging.handlers.QueueListener(_queue, *_handlers.values(), respect_handler_level=True)
    _listener.start()
    _root.addHandler(_front)


def _stop() -> None:
    global _listener
    if _listener is not None:
        _listener.stop()
        _listener = None


class _PatcherFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.__dict__.setdefault("_ctx", "")
        if _patcher is not None:
            _patcher(record)
        return True


def _formatter() -> logging.Formatter:
    return logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)


def _stream_handler(stream: TextIO, *, level: int) -> logging.Handler:
    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    handler.setFormatter(_formatter())
    return handler


logger = Logger()
"""The root logger. Bind fields onto it, or add sinks to it."""

_root.addFilter(_PatcherFilter())
_root.setLevel(TRACE)
_root.propagate = False
atexit.register(_stop)
