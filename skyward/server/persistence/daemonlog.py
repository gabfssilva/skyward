"""The daemon's own log, read back off its file and followed as it is written.

The log is a file the logger writes on a thread of its own, not a table: a read
is a scan of it, newest first, stopped by the window or the page. A scan decodes
every entry it passes, so scans run on a thread, and one at a time, for the same
reason the event log's replays take turns — they spend the interpreter lock the
event loop needs.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from datetime import datetime, timedelta
from itertools import dropwhile, islice, takewhile

from skyward.server.persistence.events import HEARTBEAT, held
from skyward.shared.observability import Entry, LogFile, Query, Summary, entries, summarize
from skyward.shared.schemas import Page

BACKLOG = 1024
"""Entries a follower may fall behind by before it is hung up on and comes back from its last id."""


class DaemonLogStore:
    """Pages, summaries and a live tail over one log file.

    A follower subscribes before it replays, so an entry written in between is held
    rather than lost, and dropped on the way out if the replay already carried it.
    The file tells a follower on the logger's thread, which only matches the entry
    and hands it to the loop; a follower too slow to keep up is hung up on, the way a
    slow reader of the event stream is, and not waited for.
    """

    def __init__(self, file: LogFile) -> None:
        self._file = file
        self._turn = asyncio.Lock()

    async def page(self, query: Query, cursor: str | None, limit: int) -> Page[Entry]:
        before = int(cursor) if cursor else None
        items = await self._scan(lambda: _page(self._file, query, before, limit))
        return Page(items=items, next_cursor=str(items[-1].sequence) if len(items) == limit else None)

    async def summary(self, query: Query, since: datetime, until: datetime, step: timedelta) -> Summary:
        sequence = self._file.sequence
        return await self._scan(lambda: summarize(entries(self._file.path), query, sequence=sequence, since=since, until=until, step=step))

    async def follow(self, query: Query, after: int | None) -> AsyncGenerator[tuple[Entry, ...], None]:
        loop = asyncio.get_running_loop()
        feed: asyncio.Queue[Entry | None] = asyncio.Queue(maxsize=BACKLOG + 1)
        hung_up = False

        def offer(entry: Entry) -> None:
            nonlocal hung_up
            if hung_up:
                return
            if feed.qsize() >= BACKLOG:
                hung_up = True
                feed.put_nowait(None)
                return
            feed.put_nowait(entry)

        def heard(entry: Entry) -> None:
            if query.matches(entry):
                loop.call_soon_threadsafe(offer, entry)

        unsubscribe = self._file.subscribe(heard)
        try:
            seen = after or 0
            if after is not None:
                replayed = await self._scan(lambda: _after(self._file, query, after))
                if replayed:
                    seen = replayed[-1].sequence
                    yield replayed

            closed = False
            while not closed:
                try:
                    async with asyncio.timeout(HEARTBEAT):
                        arrived, closed = await held(feed)
                except TimeoutError:
                    yield ()
                    continue
                run = tuple(entry for entry in arrived if entry.sequence > seen)
                if run:
                    seen = run[-1].sequence
                    yield run
        finally:
            unsubscribe()

    async def _scan[T](self, read: Callable[[], T]) -> T:
        async with self._turn:
            return await asyncio.to_thread(read)


def _page(file: LogFile, query: Query, before: int | None, limit: int) -> tuple[Entry, ...]:
    older = dropwhile(lambda entry: before is not None and entry.sequence >= before, entries(file.path))
    windowed = takewhile(lambda entry: query.since is None or entry.at >= query.since, older)
    return tuple(islice(filter(query.matches, windowed), limit))


def _after(file: LogFile, query: Query, after: int) -> tuple[Entry, ...]:
    newer = takewhile(lambda entry: entry.sequence > after, entries(file.path))
    return tuple(reversed([entry for entry in newer if query.matches(entry)]))
