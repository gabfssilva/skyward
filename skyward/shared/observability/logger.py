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
"""

from __future__ import annotations

import gzip
import logging
import logging.handlers
import os
import shutil
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TextIO

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

    def add(
        self,
        sink: str | TextIO,
        *,
        level: str = "DEBUG",
        filter: str | None = None,
        rotation: str | None = None,
        retention: int | None = None,
        compression: str | None = None,
    ) -> int:
        """Attach a sink — a path (rotating file) or a stream — and return its id.

        Parameters
        ----------
        sink
            A filesystem path, or an open text stream.
        level
            Minimum severity the sink accepts.
        filter
            Logger-name prefix the record must match.
        rotation
            Size at which a file sink rolls over, as ``"50 MB"``.
        retention
            How many rolled files to keep.
        compression
            Any truthy value gzips rolled files.
        """
        global _counter

        numeric = logging.getLevelNamesMapping().get(level.upper(), logging.DEBUG)
        match sink:
            case str() as path:
                handler = _file_handler(path, level=numeric, rotation=rotation, retention=retention, compression=compression)
            case stream:
                handler = _stream_handler(stream, level=numeric)

        if filter:
            handler.addFilter(logging.Filter(filter))

        _root.addHandler(handler)
        _counter += 1
        _handlers[_counter] = handler
        return _counter

    def remove(self, handler_id: int | None = None) -> None:
        """Detach the sink with this id, or every sink when given none."""
        if handler_id is None:
            for handler in _handlers.values():
                _root.removeHandler(handler)
            _handlers.clear()
            return
        if handler := _handlers.pop(handler_id, None):
            _root.removeHandler(handler)

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


_counter = 0
_handlers: dict[int, logging.Handler] = {}
_patcher: Patcher | None = None


class _PatcherFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.__dict__.setdefault("_ctx", "")
        if _patcher is not None:
            _patcher(record)
        return True


def _namer(name: str) -> str:
    return f"{name}.gz"


def _rotator(source: str, dest: str) -> None:
    with open(source, "rb") as raw, gzip.open(dest, "wb") as compressed:
        shutil.copyfileobj(raw, compressed)
    os.remove(source)


def _rotation_bytes(rotation: str) -> int:
    match rotation.strip().split():
        case [size, unit] if unit.upper() == "MB":
            return int(size) * 1024 * 1024
        case _:
            return 50 * 1024 * 1024


def _formatter() -> logging.Formatter:
    return logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)


def _file_handler(
    path: str,
    *,
    level: int,
    rotation: str | None,
    retention: int | None,
    compression: str | None,
) -> logging.Handler:
    handler = logging.handlers.RotatingFileHandler(
        path,
        maxBytes=_rotation_bytes(rotation) if rotation else 50 * 1024 * 1024,
        backupCount=retention if retention is not None else 10,
    )
    if compression:
        handler.namer = _namer
        handler.rotator = _rotator
    handler.setLevel(level)
    handler.setFormatter(_formatter())
    return handler


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
