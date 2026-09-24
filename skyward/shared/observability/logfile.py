"""The daemon's log on disk: one JSON object per line, numbered as it is written.

A line is an :class:`Entry` — what was said, where, by whom and about which compute —
rather than text to be taken apart again: a reader filters on the fields a writer
already had. ``sequence`` counts every entry the file has ever held and survives a
restart, so it serves as a reader's cursor and as a follower's ``Last-Event-ID``.

The file rolls over into gzipped copies beside it, and :func:`entries` reads across
them newest first. It stops at the first line that is not an entry, which is where
the history of this format begins: a file written as text before it holds nothing
here worth reading back.

A reader of a day's log is a scan, not a query: every entry is decoded and matched,
and the scan stops at the first entry older than what was asked for. At a third of a
microsecond an entry there is nothing an index would buy.
"""

from __future__ import annotations

import gzip
import logging
import logging.handlers
import os
import shutil
import threading
import traceback
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import BinaryIO, Literal

import msgspec
from msgspec import Struct

type Severity = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
"""What a file entry can be: nothing finer than ``DEBUG`` reaches the file, and a library's ``CRITICAL`` is an error."""

type Subscriber = Callable[[Entry], None]

RANK: Mapping[Severity, int] = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}


class Failure(Struct, frozen=True, kw_only=True):
    """The exception an entry was logged with.

    ``cause`` is the innermost exception, as ``Type: message``, when it is not this
    one: the first of a group's, and whatever raised it, all the way down. A market
    that could not place a machine fails as a group of groups, and what went wrong is
    at the bottom of it.
    """

    type: str
    message: str
    cause: str | None
    traceback: str


class Entry(Struct, frozen=True, kw_only=True):
    """One record, as the file holds it.

    ``site`` is the call that logged it, ``module:function:line``. ``component``,
    ``compute`` and ``node`` are the fields every reader filters on and are lifted
    out of what the logger had bound; ``fields`` is the rest of it. A record from a
    library carries its logger's top-level name as its component.
    """

    sequence: int
    at: datetime
    level: Severity
    site: str
    logger: str
    component: str | None
    compute: str | None
    node: str | None
    fields: dict[str, str]
    message: str
    exception: Failure | None

    @property
    def group(self) -> str:
        """The call that logged it, and what it was failing with: one line in a summary however often it repeats."""
        return self.site if self.exception is None else f"{self.site}|{self.exception.type}"


@dataclass(frozen=True, slots=True)
class Query:
    """The entries a reader wants. Every field narrows, and one left at its default does not.

    ``components`` and ``groups`` keep an entry with any one of theirs; ``hidden``
    drops the groups it names. ``contains`` keeps an entry that holds any one of the
    strings, ignoring case, in what it said or in the exception it carried.
    ``until`` is exclusive.
    """

    level: Severity = "DEBUG"
    components: frozenset[str] = frozenset()
    compute: str | None = None
    node: str | None = None
    groups: frozenset[str] = frozenset()
    hidden: frozenset[str] = frozenset()
    contains: tuple[str, ...] = ()
    since: datetime | None = None
    until: datetime | None = None

    def matches(self, entry: Entry) -> bool:
        return (
            RANK[entry.level] >= RANK[self.level]
            and (not self.components or entry.component in self.components)
            and (self.compute is None or entry.compute == self.compute)
            and (self.node is None or entry.node == self.node)
            and (not self.groups or entry.group in self.groups)
            and entry.group not in self.hidden
            and (self.since is None or entry.at >= self.since)
            and (self.until is None or entry.at < self.until)
            and (not self.contains or _holds(entry, self.contains))
        )


class Volume(Struct, frozen=True, kw_only=True):
    """Entries per ``step`` milliseconds from ``since``, one count per step for each level."""

    since: int
    step: int
    debug: tuple[int, ...]
    info: tuple[int, ...]
    warning: tuple[int, ...]
    error: tuple[int, ...]


class Group(Struct, frozen=True, kw_only=True):
    """The entries one call logged, failing the same way. ``series`` counts them on the volume's steps."""

    key: str
    site: str
    exception: str | None
    component: str | None
    level: Severity
    count: int
    first: datetime
    last: datetime
    computes: int
    series: tuple[int, ...]
    latest: Entry


class Summary(Struct, frozen=True, kw_only=True):
    """A window of the log at a glance.

    ``sequence`` is the newest entry the summary could have counted: a follower that
    starts from it adds to these counts without counting anything twice.
    ``components`` counts what the query would keep if it named no component, which is
    what a reader choosing among them needs to see.
    """

    sequence: int
    volume: Volume
    components: dict[str, int]
    groups: tuple[Group, ...]


class LogFile(logging.handlers.RotatingFileHandler):
    """The file, as a logging handler: it numbers what it writes and tells whoever follows it.

    It runs on the logger's one writing thread, so the numbering is the order of the
    file. A follower is told after the line is written and flushed, which is what
    lets it catch up from the file and then carry on from what it is told without a
    gap between the two. Its callback runs on that thread too, and must not block.
    """

    def __init__(self, path: Path, *, size: int = 50 * 1024 * 1024, keep: int = 10) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        super().__init__(path, maxBytes=size, backupCount=keep, encoding="utf-8")
        self.namer = _gzipped
        self.rotator = _gzip
        self.path = path
        self.sequence = next((entry.sequence for entry in entries(path)), 0)
        """The newest entry written. Everything up to it is in the file."""
        self._subscribers: tuple[Subscriber, ...] = ()
        self._subscribing = threading.Lock()

    def subscribe(self, subscriber: Subscriber) -> Callable[[], None]:
        """Be told of every entry written from now on; the call returned stops it."""
        with self._subscribing:
            self._subscribers = (*self._subscribers, subscriber)

        def unsubscribe() -> None:
            with self._subscribing:
                self._subscribers = tuple(each for each in self._subscribers if each is not subscriber)

        return unsubscribe

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = _entry(record, self.sequence + 1)
            record.__dict__["line"] = _ENCODER.encode(entry).decode()
        except Exception:
            self.handleError(record)
            return
        super().emit(record)
        self.sequence = entry.sequence
        for subscriber in self._subscribers:
            try:
                subscriber(entry)
            except Exception:
                self.handleError(record)

    def format(self, record: logging.LogRecord) -> str:
        return record.__dict__["line"]


def entries(path: Path) -> Iterator[Entry]:
    """Every entry the log file at ``path`` holds, newest first, across the files it rolled over into.

    A last line with no newline yet is still being written and is not read. The first
    line that is not an entry ends the history.
    """
    for line in _lines(path):
        try:
            yield _DECODER.decode(line)
        except msgspec.DecodeError:
            return


def failure(exc: BaseException) -> Failure:
    """What an entry says about the exception it was logged with."""
    inner = _innermost(exc)
    return Failure(
        type=_named(exc),
        message=str(exc),
        cause=None if inner is exc else f"{_named(inner)}: {inner}",
        traceback="".join(traceback.format_exception(exc)).rstrip(),
    )


def summarize(entries: Iterable[Entry], query: Query, *, sequence: int, since: datetime, until: datetime, step: timedelta) -> Summary:
    """Count, newest first, the entries between ``since`` and ``until`` that the query keeps.

    Entries newer than ``sequence`` are left for a follower to count; the scan stops at
    the first entry older than ``since``.
    """
    steps = max(1, -(-(until - since) // step))
    faceted = replace(query, components=frozenset(), since=since, until=until)
    volume = {level: [0] * steps for level in RANK}
    components: Counter[str] = Counter()
    tallies: dict[str, _Tally] = {}

    for entry in entries:
        if entry.sequence > sequence or entry.at >= until:
            continue
        if entry.at < since:
            break
        if not faceted.matches(entry):
            continue
        if entry.component:
            components[entry.component] += 1
        if query.components and entry.component not in query.components:
            continue
        index = min(steps - 1, (entry.at - since) // step)
        volume[entry.level][index] += 1
        tally = tallies.get(entry.group)
        if tally is None:
            tally = tallies[entry.group] = _Tally(latest=entry, series=[0] * steps)
        tally.add(entry, index)

    return Summary(
        sequence=sequence,
        volume=Volume(
            since=int(since.timestamp() * 1000),
            step=step // timedelta(milliseconds=1),
            debug=tuple(volume["DEBUG"]),
            info=tuple(volume["INFO"]),
            warning=tuple(volume["WARNING"]),
            error=tuple(volume["ERROR"]),
        ),
        components=dict(components),
        groups=tuple(sorted((tally.group() for tally in tallies.values()), key=lambda group: (-RANK[group.level], -group.count))),
    )


_ENCODER = msgspec.json.Encoder()
_DECODER = msgspec.json.Decoder(Entry)
_BLOCK = 1 << 20
_PROMOTED = frozenset({"component", "compute_id", "node_id"})
_OWN = "skyward.log"


@dataclass(slots=True)
class _Tally:
    latest: Entry
    series: list[int]
    count: int = 0
    level: Severity = "DEBUG"
    first: datetime | None = None
    computes: set[str] = field(default_factory=set)

    def add(self, entry: Entry, index: int) -> None:
        self.count += 1
        self.series[index] += 1
        self.first = entry.at
        self.level = max(self.level, entry.level, key=RANK.__getitem__)
        if entry.compute:
            self.computes.add(entry.compute)

    def group(self) -> Group:
        latest = self.latest
        return Group(
            key=latest.group,
            site=latest.site,
            exception=latest.exception.type if latest.exception else None,
            component=latest.component,
            level=self.level,
            count=self.count,
            first=self.first or latest.at,
            last=latest.at,
            computes=len(self.computes),
            series=tuple(self.series),
            latest=latest,
        )


def _entry(record: logging.LogRecord, sequence: int) -> Entry:
    bound: Mapping[str, object] = record.__dict__.get("extras", {})
    library = None if record.name == _OWN else record.name.split(".")[0]
    match record.__dict__.get("failure"):
        case Failure() as failed:
            exception: Failure | None = failed
        case _:
            exception = None
    return Entry(
        sequence=sequence,
        at=datetime.fromtimestamp(record.created, UTC),
        level=_severity(record.levelno),
        site=f"{record.module}:{record.funcName}:{record.lineno}",
        logger=record.name,
        component=_text(bound.get("component")) or library,
        compute=_text(bound.get("compute_id")),
        node=_text(bound.get("node_id")),
        fields={key: str(value) for key, value in bound.items() if key not in _PROMOTED},
        message=record.getMessage(),
        exception=exception,
    )


def _severity(levelno: int) -> Severity:
    if levelno >= logging.ERROR:
        return "ERROR"
    if levelno >= logging.WARNING:
        return "WARNING"
    if levelno >= logging.INFO:
        return "INFO"
    return "DEBUG"


def _text(value: object) -> str | None:
    return None if value is None else str(value)


def _holds(entry: Entry, terms: tuple[str, ...]) -> bool:
    failed = entry.exception
    said = f"{entry.message}\n{failed.type}\n{failed.message}\n{failed.cause or ''}\n{failed.traceback}" if failed else entry.message
    haystack = said.lower()
    return any(term.lower() in haystack for term in terms)


def _innermost(exc: BaseException) -> BaseException:
    match exc:
        case BaseExceptionGroup(exceptions=(first, *_)):
            return _innermost(first)
        case BaseException(__cause__=BaseException() as cause):
            return _innermost(cause)
        case _:
            return exc


def _named(exc: BaseException) -> str:
    kind = type(exc)
    return kind.__qualname__ if kind.__module__ == "builtins" else f"{kind.__module__}.{kind.__qualname__}"


def _lines(path: Path) -> Iterator[bytes]:
    if path.exists():
        yield from _backwards(path)
    index = 1
    while (rolled := path.with_name(f"{path.name}.{index}.gz")).exists():
        yield from (line for line in reversed(gzip.decompress(rolled.read_bytes()).split(b"\n")) if line)
        index += 1


def _backwards(path: Path) -> Iterator[bytes]:
    with path.open("rb") as handle:
        end = _terminated(handle)
        rest = b""
        while end > 0:
            start = max(0, end - _BLOCK)
            handle.seek(start)
            lines = (handle.read(end - start) + rest).split(b"\n")
            rest = lines.pop(0) if start > 0 else b""
            yield from (line for line in reversed(lines) if line)
            end = start


def _terminated(handle: BinaryIO) -> int:
    """Where the file's last complete line ends: past its last newline."""
    end = handle.seek(0, os.SEEK_END)
    while end > 0:
        start = max(0, end - _BLOCK)
        handle.seek(start)
        if (cut := handle.read(end - start).rfind(b"\n")) != -1:
            return start + cut + 1
        end = start
    return 0


def _gzipped(name: str) -> str:
    return f"{name}.gz"


def _gzip(source: str, dest: str) -> None:
    with open(source, "rb") as raw, gzip.open(dest, "wb", compresslevel=6) as compressed:
        shutil.copyfileobj(raw, compressed)
    os.remove(source)
