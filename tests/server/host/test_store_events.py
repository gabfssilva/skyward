"""Tests for ``Store.emit`` and ``Store.tail_events``."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from skyward.server.host.store import EventRow, Store


async def _open(tmp_path: Path) -> Store:
    store = Store(str(tmp_path / "events.db"))
    await store.open()
    return store


async def _collect(store: Store, **kwargs: object) -> list[EventRow]:
    return [row async for row in store.tail_events(**kwargs)]  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_emit_returns_monotonic_id(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        a = await store.emit("compute:x", "Pool.Started", {})
        b = await store.emit("compute:x", "Pool.Ready", {})
        c = await store.emit("compute:x", "Pool.Stopped", {})
        assert a < b < c
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_emit_persists_payload(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        await store.emit("compute:x", "Pool.Event", {"k": 1, "s": "v"})
        rows = await _collect(store)
        assert len(rows) == 1
        row = rows[0]
        assert isinstance(row, EventRow)
        assert row.aggregate == "compute:x"
        assert row.type == "Pool.Event"
        assert row.payload == {"k": 1, "s": "v"}
        assert isinstance(row.created_at, datetime)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_tail_events_since_id(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        ids = [await store.emit("compute:x", "E", {"i": i}) for i in range(5)]
        rows = await _collect(store, since_id=ids[1])
        assert [r.id for r in rows] == ids[2:]
        assert len(rows) == 3
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_tail_events_aggregate_prefix(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        await store.emit("compute:foo", "E", {})
        await store.emit("compute:bar", "E", {})
        await store.emit("node:x", "E", {})
        rows = await _collect(store, aggregate_like="compute:")
        assert {r.aggregate for r in rows} == {"compute:foo", "compute:bar"}
        assert len(rows) == 2
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_tail_terminates_at_current_tail(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        for _ in range(3):
            await store.emit("compute:x", "E", {})
        it = store.tail_events()
        rows: list[EventRow] = []
        async for row in it:
            rows.append(row)
        assert len(rows) == 3
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_emit_inside_tx_atomic_with_state(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        with pytest.raises(RuntimeError):
            async with store.tx() as tx:
                await tx.execute(
                    "INSERT INTO providers (name, type, config, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    ("p1", "aws", "{}", 0.0, 0.0),
                )
                await store.emit("provider:p1", "Provider.Created", {}, tx=tx)
                raise RuntimeError("boom")

        # Neither row persisted
        rows = await _collect(store)
        assert rows == []
        async with store.tx() as tx:
            result = await tx.fetch_all("SELECT name FROM providers")
            assert result == []
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_emit_inside_tx_commits_atomically_on_success(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        async with store.tx() as tx:
            await tx.execute(
                "INSERT INTO providers (name, type, config, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("p1", "aws", "{}", 0.0, 0.0),
            )
            eid = await store.emit("provider:p1", "Provider.Created", {"v": 1}, tx=tx)
            assert eid > 0

        rows = await _collect(store)
        assert len(rows) == 1
        assert rows[0].aggregate == "provider:p1"
        assert rows[0].payload == {"v": 1}

        async with store.tx() as tx:
            result = await tx.fetch_all("SELECT name FROM providers")
            assert [r[0] for r in result] == ["p1"]
    finally:
        await store.close()
