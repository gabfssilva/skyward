from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from msgspec import Struct

from skyward.server.persistence.store import now
from skyward.server.persistence.tables import EventRow

type Record = tuple[int, str, bytes]

BACKLOG = 1024


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
    and dropped on the way out if the replay already carried it.

    A slow consumer is disconnected, not waited for: its queue fills, its feed is
    closed, and it comes back with the sequence it got to. The alternative is a
    commit that blocks because somebody's browser tab is busy.
    """

    def __init__(self) -> None:
        self._feeds: set[asyncio.Queue[Live | None]] = set()

    async def record(self, event_type: str, payload: bytes, compute: str | None = None, task: str | None = None) -> int:
        row = EventRow(
            type=event_type,
            compute_id=compute,
            task_id=task,
            payload=payload.decode(),
            created_at=now(),
        )
        await row.save().run()

        live = Live(sequence=row.sequence, type=event_type, payload=payload, compute=compute, task=task)
        for feed in tuple(self._feeds):
            self._offer(feed, live)

        return row.sequence

    async def publish(self, event_type: str, payload: bytes, compute: str | None = None) -> None:
        """Say it once, to whoever is listening, and keep no record.

        A metric sampled every couple of seconds has no replay value, and the event
        table has no GC to save it from one. It rides the same live feed as
        :meth:`record`, without the row — a late subscriber simply misses the samples
        it was not there for, which is exactly right for a gauge.
        """
        live = Live(sequence=None, type=event_type, payload=payload, compute=compute, task=None)
        for feed in tuple(self._feeds):
            self._offer(feed, live)

    async def stream(
        self,
        last_event_id: str | None,
        compute: str | None,
        task: str | None,
        types: tuple[str, ...] | None,
    ) -> AsyncIterator[Record]:
        feed: asyncio.Queue[Live | None] = asyncio.Queue(maxsize=BACKLOG + 1)
        self._feeds.add(feed)

        try:
            seen = 0
            for record in await self._replay(int(last_event_id or 0), compute, task, types):
                seen = record[0]
                yield record

            while (live := await feed.get()) is not None:
                if not live.wanted(compute, task, types):
                    continue
                if live.sequence is None:
                    yield seen, live.type, live.payload
                elif live.sequence > seen:
                    yield live.sequence, live.type, live.payload
        finally:
            self._feeds.discard(feed)

    async def _replay(self, after: int, compute: str | None, task: str | None, types: tuple[str, ...] | None) -> list[Record]:
        query = EventRow.objects().where(EventRow.sequence > after)

        if compute:
            query = query.where(EventRow.compute_id == compute)
        if task:
            query = query.where(EventRow.task_id == task)
        if types:
            query = query.where(EventRow.type.is_in(list(types)))

        rows = await query.order_by(EventRow.sequence)
        return [(row.sequence, row.type, row.payload.encode()) for row in rows]

    def _offer(self, feed: asyncio.Queue[Live | None], live: Live) -> None:
        """Hand the event over, or hang up.

        The queue holds one slot more than the backlog, reserved for the goodbye:
        a feed that is being closed *because* it is full still has to be told, and
        a closing sentinel that itself blocks on a full queue would leave the
        consumer waiting forever on a producer that has already given up on it.
        """
        if feed.qsize() >= BACKLOG:
            self._feeds.discard(feed)
            feed.put_nowait(None)
            return

        feed.put_nowait(live)
