"""The log, and the tail of it — including the events that are never written down."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from skyward.server.persistence.db import connect
from skyward.server.persistence.events import EventStore
from skyward.server.persistence.tables import EventRow

pytestmark = pytest.mark.unit


async def test_a_published_metric_reaches_a_subscriber_but_leaves_no_row(tmp_path: Path):
    """A gauge is worth saying once, to whoever is listening now, and then forgetting."""
    await connect(tmp_path / "skyward.sqlite")
    events = EventStore()

    stream = events.stream(None, compute="cmp_1", task=None, types=None)
    received = asyncio.ensure_future(anext(stream))
    while not events._feeds:  # let the generator register its feed before the event goes out
        await asyncio.sleep(0)

    await events.publish("node.metrics", b'{"name":"gpu_util","value":87.5}', compute="cmp_1")

    _, event_type, payload = await asyncio.wait_for(received, timeout=1)
    await stream.aclose()

    assert event_type == "node.metrics"
    assert payload == b'{"name":"gpu_util","value":87.5}'
    assert await EventRow.objects() == [], "a transient event writes no row"
