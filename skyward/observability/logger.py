"""Drop-in replacement for loguru backed by stdlib logging + rich.

Usage (identical to loguru)::

    from skyward.observability.logger import logger

    log = logger.bind(component="pool")
    log.info("Started {n} nodes", n=4)
"""

from __future__ import annotations

import gzip
import inspect
import logging
import logging.handlers
import os
import shutil
import sys
from collections.abc import Callable
from typing import TextIO

from rich.logging import RichHandler

TRACE = 5
logging.addLevelName(TRACE, "TRACE")

_skyward_root = logging.getLogger("skyward")

type Patcher = Callable[[logging.LogRecord], None]


def _caller_logger() -> logging.Logger:
    frame = inspect.stack()[2]
    module = frame.frame.f_globals.get("__name__", "skyward")
    return logging.getLogger(module)


def _format_message(msg: str, args: tuple[object, ...], kwargs: dict[str, object]) -> str:
    if kwargs:
        return msg.format(**kwargs)
    if args:
        return msg.format(*args)
    return msg


class BoundLogger:
    __slots__ = ("_extras",)

    def __init__(self, extras: dict[str, object] | None = None) -> None:
        self._extras = extras or {}

    def bind(self, **kwargs: object) -> BoundLogger:
        return BoundLogger({**self._extras, **kwargs})

    def _log(self, level: int, message: str, /, *args: object, **kwargs: object) -> None:
        exc_info = kwargs.pop("exc_info", False)
        text = _format_message(message, args, kwargs)
        lib_logger = _caller_logger()
        if not lib_logger.isEnabledFor(level):
            return
        record = lib_logger.makeRecord(
            name=lib_logger.name,
            level=level,
            fn="",
            lno=0,
            msg=text,
            args=(),
            exc_info=None,
        )
        frame = inspect.stack()[2]
        record.pathname = frame.filename
        record.filename = os.path.basename(frame.filename)
        record.lineno = frame.lineno
        record.funcName = frame.function
        for k, v in self._extras.items():
            setattr(record, k, v)
        record.extras = self._extras  # type: ignore[attr-defined]
        if exc_info:
            record.exc_info = sys.exc_info()
        lib_logger.handle(record)

    def trace(self, message: str, /, *args: object, **kwargs: object) -> None:
        self._log(TRACE, message, *args, **kwargs)

    def debug(self, message: str, /, *args: object, **kwargs: object) -> None:
        self._log(logging.DEBUG, message, *args, **kwargs)

    def info(self, message: str, /, *args: object, **kwargs: object) -> None:
        self._log(logging.INFO, message, *args, **kwargs)

    def warning(self, message: str, /, *args: object, **kwargs: object) -> None:
        self._log(logging.WARNING, message, *args, **kwargs)

    def error(self, message: str, /, *args: object, **kwargs: object) -> None:
        self._log(logging.ERROR, message, *args, **kwargs)

    def exception(self, message: str, /, *args: object, **kwargs: object) -> None:
        kwargs["exc_info"] = True
        self._log(logging.ERROR, message, *args, **kwargs)


_handler_counter = 0
_handlers: dict[int, logging.Handler] = {}
_patcher: Patcher | None = None


class _PatcherFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if _patcher is not None:
            _patcher(record)
        return True


_patcher_filter = _PatcherFilter()
_skyward_root.addFilter(_patcher_filter)


def _namer(name: str) -> str:
    return name + ".gz"


def _rotator(source: str, dest: str) -> None:
    with open(source, "rb") as f_in, gzip.open(dest, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(source)


def _parse_rotation_bytes(rotation: str) -> int:
    parts = rotation.strip().split()
    match parts:
        case [num, unit] if unit.upper() == "MB":
            return int(num) * 1024 * 1024
        case _:
            return 50 * 1024 * 1024


def _make_file_handler(
    path: str,
    *,
    level: int,
    rotation: str | None,
    retention: int | None,
    compression: str | None,
) -> logging.Handler:
    max_bytes = _parse_rotation_bytes(rotation) if rotation else 50 * 1024 * 1024
    backup_count = retention if retention is not None else 10
    handler = logging.handlers.RotatingFileHandler(
        path,
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    if compression:
        handler.namer = _namer
        handler.rotator = _rotator
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d | %(levelname)-8s | "
        "%(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    return handler


def _make_console_handler(level: int) -> logging.Handler:
    handler = RichHandler(
        level=level,
        console=None,
        show_time=True,
        show_level=True,
        show_path=True,
        markup=True,
        rich_tracebacks=True,
    )
    handler.setLevel(level)
    return handler


class LoguruCompat:
    def __init__(self) -> None:
        self._bound = BoundLogger()

    def bind(self, **kwargs: object) -> BoundLogger:
        return self._bound.bind(**kwargs)

    def trace(self, message: str, /, *args: object, **kwargs: object) -> None:
        self._bound.trace(message, *args, **kwargs)

    def debug(self, message: str, /, *args: object, **kwargs: object) -> None:
        self._bound.debug(message, *args, **kwargs)

    def info(self, message: str, /, *args: object, **kwargs: object) -> None:
        self._bound.info(message, *args, **kwargs)

    def warning(self, message: str, /, *args: object, **kwargs: object) -> None:
        self._bound.warning(message, *args, **kwargs)

    def error(self, message: str, /, *args: object, **kwargs: object) -> None:
        self._bound.error(message, *args, **kwargs)

    def exception(self, message: str, /, *args: object, **kwargs: object) -> None:
        self._bound.exception(message, *args, **kwargs)

    def remove(self, handler_id: int | None = None) -> None:
        if handler_id is None:
            for h in list(_handlers.values()):
                _skyward_root.removeHandler(h)
            _handlers.clear()
            return
        if h := _handlers.pop(handler_id, None):
            _skyward_root.removeHandler(h)

    def add(
        self,
        sink: str | TextIO,
        *,
        level: str = "DEBUG",
        filter: str | None = None,  # noqa: A002
        rotation: str | None = None,
        retention: int | None = None,
        compression: str | None = None,
        **_kwargs: object,
    ) -> int:
        global _handler_counter
        numeric_level = getattr(logging, level.upper(), logging.DEBUG)

        match sink:
            case str() as path:
                handler = _make_file_handler(
                    path,
                    level=numeric_level,
                    rotation=rotation,
                    retention=retention,
                    compression=compression,
                )
            case _:
                handler = _make_console_handler(numeric_level)

        if filter:
            handler.addFilter(logging.Filter(filter))

        _skyward_root.addHandler(handler)
        _handler_counter += 1
        _handlers[_handler_counter] = handler
        return _handler_counter

    def enable(self, name: str) -> None:
        target = logging.getLogger(name)
        target.disabled = False
        target.setLevel(TRACE)

    def disable(self, name: str) -> None:
        logging.getLogger(name).disabled = True

    def configure(self, *, patcher: Patcher | None = None) -> None:
        global _patcher
        if patcher is not None:
            _patcher = patcher


logger = LoguruCompat()

_skyward_root.setLevel(TRACE)
_skyward_root.propagate = False
