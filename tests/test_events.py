"""The event log's tail: replays handed over a page at a time, filtered feeds, cursors on published frames, batched records, hang-ups."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import msgspec
import pytest

from skyward.server.persistence.db import connect
from skyward.server.persistence.events import BACKLOG, PAGE, EventStore, Live, Record
from skyward.server.persistence.store import now
from skyward.server.persistence.tables import EventRow
from skyward.shared.events import ConsoleEvent, LogEntry, MetricEvent, NodeEvent
from skyward.shared.schemas import Page

pytestmark = pytest.mark.local

WAIT = 5.0

type Runs = AsyncIterator[tuple[Record, ...]]


def line(compute: str, content: str, node: str = "n0") -> ConsoleEvent:
    return ConsoleEvent(compute=compute, node=node, content=content)


def content(payload: bytes) -> str:
    return msgspec.json.decode(payload, type=ConsoleEvent).content


async def outgrown(compute: str) -> None:
    """A row as an older release wrote it: the payload names neither its own type nor the state it moved to."""
    await EventRow.raw(
        "INSERT INTO events (type, compute_id, payload, created_at) VALUES ('node.requested', {}, {}, {})",
        compute,
        f'{{"compute":"{compute}","node":"n0"}}',
        now(),
    ).run()


async def run(stream: Runs) -> tuple[Record, ...]:
    return await asyncio.wait_for(anext(stream), WAIT)


async def take(stream: Runs, count: int) -> list[Record]:
    """The next ``count`` records, however the stream grouped them."""
    records: list[Record] = []
    while len(records) < count:
        records.extend(await run(stream))
    assert len(records) == count
    return records


@pytest.fixture
async def events(tmp_path: Path) -> EventStore:
    await connect(tmp_path / "events.db")
    return EventStore()


def describe_replay() -> None:
    async def it_delivers_more_than_one_page_complete_and_in_order(events: EventStore) -> None:
        total = PAGE * 2 + 200
        await events.record_all([line("a", str(index)) for index in range(total)])

        stream = events.stream(None, None, None, None)
        try:
            records = await take(stream, total)
        finally:
            await stream.aclose()

        sequences = [sequence for sequence, _, _ in records]
        assert sequences == sorted(set(sequences))
        assert [content(payload) for _, _, payload in records] == [str(index) for index in range(total)]

    async def it_hands_each_page_over_whole(events: EventStore) -> None:
        await events.record_all([line("a", str(index)) for index in range(PAGE * 2 + 200)])

        stream = events.stream(None, None, None, None)
        try:
            sizes = [len(await run(stream)) for _ in range(3)]
        finally:
            await stream.aclose()

        assert sizes == [PAGE, PAGE, 200]

    async def it_is_not_held_up_by_a_subscriber_that_stopped_reading(events: EventStore) -> None:
        total = PAGE * 3
        await events.record_all([line("a", str(index)) for index in range(total)])

        stalled = events.stream(None, None, None, None)
        reading = events.stream(None, None, None, None)
        try:
            await run(stalled)
            assert len(await take(reading, total)) == total
        finally:
            await stalled.aclose()
            await reading.aclose()


def describe_filtered_feed() -> None:
    async def it_is_neither_woken_nor_disconnected_by_another_computes_events(events: EventStore) -> None:
        await events.record(line("a", "first"))
        stream = events.stream(None, "a", None, None)
        try:
            assert [content(payload) for _, _, payload in await run(stream)] == ["first"]

            waiting = asyncio.ensure_future(anext(stream))
            await asyncio.sleep(0.01)
            noise = msgspec.json.encode(line("b", "noise"))
            for sequence in range(1, BACKLOG * 3):
                events.deliver(Live(sequence=10_000 + sequence, type="node.console", payload=noise, compute="b", task=None))
            await events.record(line("b", "persisted noise"))
            await asyncio.sleep(0.05)
            assert not waiting.done()

            await events.record(line("a", "second"))
            second = await asyncio.wait_for(waiting, WAIT)
            assert [content(payload) for _, _, payload in second] == ["second"]
        finally:
            await stream.aclose()


def describe_published_frame() -> None:
    async def it_carries_the_resume_cursor_when_nothing_is_replayed(events: EventStore) -> None:
        for index in range(3):
            await events.record(line("a", str(index)))
        stream = events.stream(None, None, None, None)
        try:
            cursor = (await take(stream, 3))[-1][0]
        finally:
            await stream.aclose()

        resumed = events.stream(str(cursor), None, None, None)
        try:
            waiting = asyncio.ensure_future(anext(resumed))
            await asyncio.sleep(0.05)
            await events.publish(MetricEvent(compute="a", node="n0", name="cpu", value=0.5))
            ((sequence, frame, _),) = await asyncio.wait_for(waiting, WAIT)
        finally:
            await resumed.aclose()

        assert frame == "node.metrics"
        assert sequence == cursor

    async def it_carries_the_last_persisted_event_it_was_sent(events: EventStore) -> None:
        await events.record(line("a", "replayed"))
        stream = events.stream(None, None, None, None)
        try:
            await take(stream, 1)
            await events.record(line("a", "live"))
            ((live, _, _),) = await run(stream)
            await events.publish(MetricEvent(compute="a", node="n0", name="cpu", value=0.5))
            ((sequence, frame, _),) = await run(stream)
        finally:
            await stream.aclose()

        assert frame == "node.metrics"
        assert sequence == live

    async def it_carries_the_record_before_it_in_the_same_run(events: EventStore) -> None:
        await events.record(line("a", "replayed"))
        stream = events.stream(None, None, None, None)
        try:
            await take(stream, 1)
            await events.record_all([line("a", "one"), line("a", "two")])
            await events.publish(MetricEvent(compute="a", node="n0", name="cpu", value=0.5))
            together = await run(stream)
        finally:
            await stream.aclose()

        assert [frame for _, frame, _ in together] == ["node.console", "node.console", "node.metrics"]
        assert together[2][0] == together[1][0]


def describe_record_all() -> None:
    async def it_stores_and_delivers_several_events_in_the_order_given(events: EventStore) -> None:
        await events.record(line("a", "before"))
        stream = events.stream(None, None, None, None)
        try:
            ((before, _, _),) = await run(stream)
            given = ["z", "m", "a", "q"]
            await events.record_all([line("a", text) for text in given])
            delivered = await take(stream, len(given))
        finally:
            await stream.aclose()

        sequences = [sequence for sequence, _, _ in delivered]
        assert [content(payload) for _, _, payload in delivered] == given
        assert sequences == sorted(sequences) and len(set(sequences)) == len(given) and sequences[0] > before

        replay = events.stream(str(before), None, None, None)
        try:
            stored = await take(replay, len(given))
        finally:
            await replay.aclose()

        assert [(sequence, content(payload)) for sequence, _, payload in stored] == list(zip(sequences, given, strict=True))

    async def it_does_nothing_with_no_events(events: EventStore) -> None:
        await events.record_all([])
        await events.record(line("a", "only"))
        stream = events.stream(None, None, None, None)
        try:
            assert [content(payload) for _, _, payload in await run(stream)] == ["only"]
        finally:
            await stream.aclose()


def describe_a_feed_hung_up_on() -> None:
    async def it_still_hands_over_what_it_held_and_then_ends(events: EventStore) -> None:
        stream = events.stream(None, "a", None, None)
        try:
            waiting = asyncio.ensure_future(anext(stream))
            await asyncio.sleep(0.05)
            flood = msgspec.json.encode(line("a", "flood"))
            for sequence in range(1, BACKLOG + 6):
                events.deliver(Live(sequence=sequence, type="node.console", payload=flood, compute="a", task=None))
            held = await asyncio.wait_for(waiting, WAIT)
            with pytest.raises(StopAsyncIteration):
                await run(stream)
        finally:
            await stream.aclose()

        assert [sequence for sequence, _, _ in held] == list(range(1, BACKLOG + 1))


def describe_reading_the_log() -> None:
    async def it_reads_the_newest_first_a_page_at_a_time(events: EventStore) -> None:
        await events.record_all([line("cmp_a", f"line {index}") for index in range(5)])

        first = await events.log(None, 2)
        second = await events.log(first.next_cursor, 2)
        last = await events.log(second.next_cursor, 2)

        assert [*said(first), *said(second), *said(last)] == ["line 4", "line 3", "line 2", "line 1", "line 0"]
        assert last.next_cursor is None

    async def it_reads_one_computes_events_among_the_others(events: EventStore) -> None:
        await events.record_all([line("cmp_a", "a0"), line("cmp_b", "b0"), line("cmp_a", "a1")])

        page = await events.log(None, 10, compute="cmp_a")

        assert said(page) == ["a1", "a0"]

    async def it_reads_only_the_frames_asked_for(events: EventStore) -> None:
        await events.record(NodeEvent(compute="cmp_a", node="n0", state="ready"))
        await events.record(line("cmp_a", "printed"))

        page = await events.log(None, 10, types=("node.ready",))

        assert [entry.type for entry in page.items] == ["node.ready"]

    async def it_reads_one_nodes_lines_among_the_others(events: EventStore) -> None:
        await events.record_all([line("cmp_a", "from zero", "n0"), line("cmp_a", "from one", "n1")])

        page = await events.log(None, 10, node="n1")

        assert said(page) == ["from one"]

    async def it_reads_only_the_lines_that_said_one_of_the_words(events: EventStore) -> None:
        await events.record_all([line("cmp_a", "CUDA out of memory"), line("cmp_a", "epoch 3 done"), line("cmp_a", "Traceback (most recent call last)")])

        page = await events.log(None, 10, contains=("traceback", "out of memory"))

        assert said(page) == ["Traceback (most recent call last)", "CUDA out of memory"]

    async def a_search_answers_with_lines_rather_than_with_payloads(events: EventStore) -> None:
        await events.record(NodeEvent(compute="cmp_a", node="n0", state="ready"))

        page = await events.log(None, 10, contains=("ready",))

        assert page.items == (), "the word is in the row, not in anything anybody printed"

    async def a_row_this_daemon_cannot_read_is_skipped_rather_than_failing_the_page(events: EventStore) -> None:
        await events.record(line("cmp_a", "before"))
        await outgrown("cmp_a")
        await events.record(line("cmp_a", "after"))

        page = await events.log(None, 10)

        assert said(page) == ["after", "before"]

    async def a_page_of_rows_it_cannot_read_still_leads_to_the_next_one(events: EventStore) -> None:
        await events.record(line("cmp_a", "oldest"))
        await outgrown("cmp_a")
        await outgrown("cmp_a")
        await events.record(line("cmp_a", "newest"))

        first = await events.log(None, 3)
        second = await events.log(first.next_cursor, 3)

        assert said(first) == ["newest"] and first.next_cursor is not None, "the cursor is a row, not an entry"
        assert said(second) == ["oldest"]

    async def it_says_when_each_event_was_recorded(events: EventStore) -> None:
        before = now()
        await events.record(line("cmp_a", "printed"))

        (entry,) = (await events.log(None, 10)).items

        assert before <= entry.at <= now()

    async def a_page_is_followed_by_the_stream_from_its_newest_entry(events: EventStore) -> None:
        await events.record(line("cmp_a", "before the page"))
        (newest,) = (await events.log(None, 1)).items
        await events.record(line("cmp_a", "after the page"))

        (record,) = await take(events.stream(str(newest.sequence), None, None, None), 1)

        assert content(record[2]) == "after the page"


def said(page: Page[LogEntry]) -> list[str]:
    return [entry.data.content for entry in page.items if isinstance(entry.data, ConsoleEvent)]
