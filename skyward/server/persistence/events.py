from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence

from msgspec import Struct

from skyward.server.persistence.store import now
from skyward.server.persistence.tables import EventRow
from skyward.shared import codec
from skyward.shared.events import ConsoleEvent, Event, TaskEvent, name

type Record = tuple[int, str, bytes]
type Filter = tuple[str | None, str | None, tuple[str, ...] | None]

BACKLOG = 1024

PAGE = 500
"""Rows one replay query reads: a long backlog is paged, not loaded whole."""


class Live(Struct, frozen=True):
    sequence: int | None
    type: str
    payload: bytes
    compute: str | None
    task: str | None

    def wanted(self, compute: str | None, task: str | None, types: tuple[str, ...] | None) -> bool:
        return not (
            (types and self.type not in types)
            or (compute and self.compute != compute)
            or (task and self.task != task)
        )


class EventStore:
    """The log, and the tail of it.

    A subscriber replays from the table and then hangs off a live feed; nothing
    here polls for new rows. The two are stitched by subscribing *before* the
    replay reads, so an event committed in between is buffered rather than lost,
    and dropped on the way out if the replay already carried it. The replay reads
    in pages of :data:`PAGE` rows, and a feed is only handed the events its
    filter wants, so a subscriber is neither woken nor filled by the rest.

    Records are handed over in runs, not one by one: a replay page as it came off
    the table, or everything a feed held when its subscriber woke. Most of what a
    record costs on its way out is the write that carries it, and a run goes out
    in one.

    Replay pages are read one at a time, whoever asks. Every row crosses into
    Python with a hand-off of the interpreter lock, and connections reading side
    by side spend more handing it over than reading: replays that take turns
    finish sooner than replays that race, and the loop, which needs the same lock
    for everything else the daemon does, stops waiting behind them.

    A slow consumer is disconnected, not waited for: its queue fills, its feed is
    closed, and it comes back with the sequence it got to. The alternative is a
    commit that blocks because somebody's browser tab is busy.
    """

    def __init__(self) -> None:
        self._feeds: dict[asyncio.Queue[Live | None], Filter] = {}
        self._turn = asyncio.Lock()

    async def record(self, event: Event) -> None:
        """Write it down and hand it to whoever is listening, in that order."""
        self.deliver(await self.append(event))

    async def record_all(self, events: Sequence[Event]) -> None:
        """Write the lines a node printed together in one statement, then hand them over in order.

        A single INSERT is atomic, and SQLite assigns the rowids of its rows in the
        order they were given, so the sequences sorted line up with the events.
        """
        if not events:
            return

        rows = [await _row(event) for event in events]
        inserted = await EventRow.insert(*(row for row, _, _ in rows)).returning(EventRow.sequence).run()
        sequences = sorted(item["sequence"] for item in inserted)

        for (_, frame, payload), sequence, event in zip(rows, sequences, events, strict=True):
            self.deliver(Live(sequence=sequence, type=frame, payload=payload, compute=event.compute, task=_task(event)))

    async def append(self, event: Event) -> Live:
        """Write the row and say nothing yet.

        The half of :meth:`record` that belongs inside a transaction. What comes back
        is handed to :meth:`deliver` once the transaction has committed — a
        subscriber told of an event that is then rolled back has been told a lie the
        replay will never repeat, and a cursor past it would skip what did happen.

        The frame name and the filter columns are the event's own: every event names
        its compute, and the ones about a task's execution or a task name that too.
        """
        row, frame, payload = await _row(event)
        await row.save().run()
        return Live(sequence=row.sequence, type=frame, payload=payload, compute=event.compute, task=_task(event))

    def deliver(self, live: Live) -> None:
        """Hand a committed event to whoever is listening."""
        for feed, (compute, task, types) in tuple(self._feeds.items()):
            if live.wanted(compute, task, types):
                self._offer(feed, live)

    async def publish(self, event: Event) -> None:
        """Say it once, to whoever is listening, and keep no record.

        A metric sampled every couple of seconds has no replay value, and the event
        table has no GC to save it from one. It rides the same live feed as
        :meth:`record`, without the row — a late subscriber simply misses the samples
        it was not there for, which is exactly right for a gauge.
        """
        payload = await codec.json(Event).encode(event)
        self.deliver(Live(sequence=None, type=name(event), payload=payload, compute=event.compute, task=None))

    async def stream(
        self,
        last_event_id: str | None,
        compute: str | None,
        task: str | None,
        types: tuple[str, ...] | None,
    ) -> AsyncIterator[tuple[Record, ...]]:
        feed: asyncio.Queue[Live | None] = asyncio.Queue(maxsize=BACKLOG + 1)
        self._feeds[feed] = (compute, task, types)

        try:
            cursor = int(last_event_id or 0)
            seen = 0
            async for page in self._replay(cursor, compute, task, types):
                seen = page[-1][0]
                cursor = max(cursor, seen)
                yield page

            closed = False
            while not closed:
                held, closed = await _held(feed)
                run: list[Record] = []
                for live in held:
                    if live.sequence is None:
                        run.append((cursor, live.type, live.payload))
                    elif live.sequence > seen:
                        cursor = max(cursor, live.sequence)
                        run.append((live.sequence, live.type, live.payload))
                if run:
                    yield tuple(run)
        finally:
            self._feeds.pop(feed, None)

    async def _replay(self, after: int, compute: str | None, task: str | None, types: tuple[str, ...] | None) -> AsyncIterator[tuple[Record, ...]]:
        while True:
            query = EventRow.select(EventRow.sequence, EventRow.type, EventRow.payload).where(EventRow.sequence > after)

            if compute:
                query = query.where(EventRow.compute_id == compute)
            if task:
                query = query.where(EventRow.task_id == task)
            if types:
                query = query.where(EventRow.type.is_in(list(types)))

            async with self._turn:
                rows = await query.order_by(EventRow.sequence).limit(PAGE)
            if rows:
                yield tuple((row["sequence"], row["type"], row["payload"].encode()) for row in rows)

            if len(rows) < PAGE:
                return
            after = rows[-1]["sequence"]

    def _offer(self, feed: asyncio.Queue[Live | None], live: Live) -> None:
        """Hand the event over, or hang up.

        The queue holds one slot more than the backlog, reserved for the goodbye:
        a feed that is being closed *because* it is full still has to be told, and
        a closing sentinel that itself blocks on a full queue would leave the
        consumer waiting forever on a producer that has already given up on it.
        """
        if feed.qsize() >= BACKLOG:
            self._feeds.pop(feed, None)
            feed.put_nowait(None)
            return

        feed.put_nowait(live)


async def _held(feed: asyncio.Queue[Live | None]) -> tuple[list[Live], bool]:
    """Wait for the next event, then take every other one the feed already holds.

    The flag says the feed was hung up on. The goodbye is queued behind the events
    that came before it, and those are still handed over.
    """
    held: list[Live] = []
    item = await feed.get()
    while item is not None:
        held.append(item)
        if feed.empty():
            return held, False
        item = feed.get_nowait()
    return held, True


async def _row(event: Event) -> tuple[EventRow, str, bytes]:
    frame = name(event)
    payload = await codec.json(Event).encode(event)
    row = EventRow(
        type=frame,
        compute_id=event.compute,
        task_id=_task(event),
        payload=payload.decode(),
        created_at=now(),
    )
    return row, frame, payload


def _task(event: Event) -> str | None:
    match event:
        case TaskEvent(task=task) | ConsoleEvent(task=task):
            return task
        case _:
            return None
