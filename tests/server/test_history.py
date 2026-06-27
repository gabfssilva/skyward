"""Tests for the server-side execution history store."""

from __future__ import annotations

import pytest

pytest.importorskip("starlette")

from skyward.api.events import Log  # noqa: E402
from skyward.server.history import HistoryStore  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


def _log(pool: str, tid: str, msg: str) -> Log.Emitted:
    return Log.Emitted(pool_name=pool, node_id=0, message=msg, task_id=tid)


def test_append_and_get():
    store = HistoryStore()
    store.append(_log("p", "t1", "hello"))
    events = store.get("p")
    assert len(events) == 1
    assert events[0]["type"] == "Log.Emitted"
    assert events[0]["fields"]["message"] == "hello"


def test_get_limit():
    store = HistoryStore()
    for i in range(5):
        store.append(_log("p", "t", f"m{i}"))
    last2 = store.get("p", 2)
    assert [e["fields"]["message"] for e in last2] == ["m3", "m4"]


def test_ring_eviction():
    store = HistoryStore(capacity=3)
    for i in range(5):
        store.append(_log("p", "t", f"m{i}"))
    msgs = [e["fields"]["message"] for e in store.get("p")]
    assert msgs == ["m2", "m3", "m4"]


def test_ignores_event_without_pool_name():
    store = HistoryStore()

    class _NoPool:
        node_id = 0

    store.append(_NoPool())  # type: ignore[arg-type]
    assert store.get("p") == []
    assert store.has("p") is False


def test_sources_round_trip():
    store = HistoryStore()
    store.set_source("p", "eid1", "print('hi')")
    assert store.sources("p") == {"eid1": "print('hi')"}
    assert store.has("p") is True


def test_drop():
    store = HistoryStore()
    store.append(_log("p", "t", "x"))
    store.set_source("p", "e", "src")
    store.drop("p")
    assert store.get("p") == []
    assert store.sources("p") == {}
    assert store.has("p") is False
