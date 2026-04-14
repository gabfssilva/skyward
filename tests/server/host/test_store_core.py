"""Tests for ``Store`` lifecycle and transaction context."""

import asyncio
from pathlib import Path

import aiosqlite
import pytest

from skyward.server.host.store import Store


@pytest.mark.asyncio
async def test_open_applies_schema_and_sets_wal(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    store = Store(str(db_path))
    await store.open()
    try:
        async with aiosqlite.connect(db_path) as probe:
            cursor = await probe.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            rows = await cursor.fetchall()
            names = {row[0] for row in rows} - {"sqlite_sequence"}
            assert "compute" in names
            assert "events" in names
            assert "providers" in names

            cursor = await probe.execute("PRAGMA journal_mode")
            row = await cursor.fetchone()
            assert row is not None
            assert row[0].lower() == "wal"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_close_closes_connections(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    store = Store(str(db_path))
    await store.open()
    await store.close()
    with pytest.raises(Exception):
        async with store.tx() as tx:
            await tx.execute("SELECT 1")


@pytest.mark.asyncio
async def test_tx_commits_on_exit(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    store = Store(str(db_path))
    await store.open()
    try:
        async with store.tx() as tx:
            await tx.execute(
                "INSERT INTO providers(name, type, config, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("aws-1", "aws", "{}", 1.0, 1.0),
            )

        async with aiosqlite.connect(db_path) as probe:
            cursor = await probe.execute(
                "SELECT name FROM providers WHERE name=?", ("aws-1",)
            )
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == "aws-1"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_tx_rolls_back_on_exception(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    store = Store(str(db_path))
    await store.open()
    try:
        with pytest.raises(RuntimeError, match="boom"):
            async with store.tx() as tx:
                await tx.execute(
                    "INSERT INTO providers(name, type, config, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    ("aws-rollback", "aws", "{}", 1.0, 1.0),
                )
                raise RuntimeError("boom")

        async with aiosqlite.connect(db_path) as probe:
            cursor = await probe.execute(
                "SELECT name FROM providers WHERE name=?", ("aws-rollback",)
            )
            row = await cursor.fetchone()
            assert row is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_tx_atomicity_no_partial_writes(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    store = Store(str(db_path))
    await store.open()
    try:
        with pytest.raises(aiosqlite.Error):
            async with store.tx() as tx:
                await tx.execute(
                    "INSERT INTO providers(name, type, config, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    ("atomic", "aws", "{}", 1.0, 1.0),
                )
                await tx.execute(
                    "INSERT INTO events(ts, aggregate, type, payload) "
                    "VALUES (?, ?, ?, ?)",
                    (1.0, "provider:atomic", "created", "not-valid-json"),
                )

        async with aiosqlite.connect(db_path) as probe:
            cursor = await probe.execute(
                "SELECT count(*) FROM providers WHERE name=?", ("atomic",)
            )
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == 0

            cursor = await probe.execute(
                "SELECT count(*) FROM events WHERE aggregate=?", ("provider:atomic",)
            )
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == 0
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_writer_is_serialized(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    store = Store(str(db_path))
    await store.open()
    try:
        async def insert(i: int) -> None:
            async with store.tx() as tx:
                await tx.execute(
                    "INSERT INTO providers(name, type, config, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (f"p-{i}", "aws", "{}", float(i), float(i)),
                )

        await asyncio.gather(*[insert(i) for i in range(10)])

        async with aiosqlite.connect(db_path) as probe:
            cursor = await probe.execute("SELECT count(*) FROM providers")
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == 10
    finally:
        await store.close()
