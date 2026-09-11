"""The event log's tail: paged replays, filtered feeds, cursors on published frames, batched records."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import msgspec
import pytest

from skyward.server.persistence.db import connect
from skyward.server.persistence.events import BACKLOG, PAGE, EventStore, Live
from skyward.shared.events import ConsoleEvent, MetricEvent

pytestmark = pytest.mark.local

WAIT = 5.0


def line(compute: str, content: str) -> ConsoleEvent:
    return ConsoleEvent(compute=compute, node="n0", content=content)


def content(payload: bytes) -> str:
    return msgspec.json.decode(payload, type=ConsoleEvent).content


async def pull(stream: AsyncIterator[tuple[int, str, bytes]]) -> tuple[int, str, bytes]:
    return await asyncio.wait_for(anext(stream), WAIT)


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
            records = [await pull(stream) for _ in range(total)]
        finally:
            await stream.aclose()

        sequences = [sequence for sequence, _, _ in records]
        assert sequences == sorted(set(sequences))
        assert [content(payload) for _, _, payload in records] == [str(index) for index in range(total)]


def describe_filtered_feed() -> None:
    async def it_is_neither_woken_nor_disconnected_by_another_computes_events(events: EventStore) -> None:
        await events.record(line("a", "first"))
        stream = events.stream(None, "a", None, None)
        try:
            assert content((await pull(stream))[2]) == "first"

            waiting = asyncio.ensure_future(anext(stream))
            await asyncio.sleep(0.01)
            payload = msgspec.json.encode(line("b", "noise"))
            for sequence in range(1, BACKLOG * 3):
                events.deliver(Live(sequence=10_000 + sequence, type="node.console", payload=payload, compute="b", task=None))
            await events.record(line("b", "persisted noise"))
            await asyncio.sleep(0.05)
            assert not waiting.done()

            await events.record(line("a", "second"))
            _, _, second = await asyncio.wait_for(waiting, WAIT)
            assert content(second) == "second"
        finally:
            await stream.aclose()


def describe_published_frame() -> None:
    async def it_carries_the_resume_cursor_when_nothing_is_replayed(events: EventStore) -> None:
        for index in range(3):
            await events.record(line("a", str(index)))
        stream = events.stream(None, None, None, None)
        try:
            cursor = [await pull(stream) for _ in range(3)][-1][0]
        finally:
            await stream.aclose()

        resumed = events.stream(str(cursor), None, None, None)
        try:
            waiting = asyncio.ensure_future(anext(resumed))
            await asyncio.sleep(0.05)
            await events.publish(MetricEvent(compute="a", node="n0", name="cpu", value=0.5))
            sequence, frame, _ = await asyncio.wait_for(waiting, WAIT)
        finally:
            await resumed.aclose()

        assert frame == "node.metrics"
        assert sequence == cursor

    async def it_carries_the_last_persisted_event_it_was_sent(events: EventStore) -> None:
        await events.record(line("a", "replayed"))
        stream = events.stream(None, None, None, None)
        try:
            await pull(stream)
            await events.record(line("a", "live"))
            live, _, _ = await pull(stream)
            await events.publish(MetricEvent(compute="a", node="n0", name="cpu", value=0.5))
            sequence, frame, _ = await pull(stream)
        finally:
            await stream.aclose()

        assert frame == "node.metrics"
        assert sequence == live


def describe_record_all() -> None:
    async def it_stores_and_delivers_several_events_in_the_order_given(events: EventStore) -> None:
        await events.record(line("a", "before"))
        stream = events.stream(None, None, None, None)
        try:
            before, _, _ = await pull(stream)
            given = ["z", "m", "a", "q"]
            await events.record_all([line("a", text) for text in given])
            delivered = [await pull(stream) for _ in given]
        finally:
            await stream.aclose()

        sequences = [sequence for sequence, _, _ in delivered]
        assert [content(payload) for _, _, payload in delivered] == given
        assert sequences == sorted(sequences) and len(set(sequences)) == len(given) and sequences[0] > before

        replay = events.stream(str(before), None, None, None)
        try:
            stored = [await pull(replay) for _ in given]
        finally:
            await replay.aclose()

        assert [(sequence, content(payload)) for sequence, _, payload in stored] == list(zip(sequences, given, strict=True))

    async def it_does_nothing_with_no_events(events: EventStore) -> None:
        await events.record_all([])
        await events.record(line("a", "only"))
        stream = events.stream(None, None, None, None)
        try:
            assert content((await pull(stream))[2]) == "only"
        finally:
            await stream.aclose()
