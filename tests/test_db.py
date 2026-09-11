"""The store's connection pool, under the failures it exists to absorb."""

import asyncio
import sqlite3
from pathlib import Path

import pytest

from skyward.server.persistence import db
from skyward.server.persistence.db import POOL_SIZE, connect
from skyward.server.persistence.tables import ComputeRow

pytestmark = pytest.mark.local


def describe_the_connection_pool() -> None:
    async def a_connection_that_fails_to_open_gives_its_slot_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """More failed opens than the pool has slots, and the next query still runs."""
        await connect(tmp_path / "skyward.sqlite")
        engine = db._current
        assert engine is not None
        engine.dispose()
        for _ in range(POOL_SIZE):
            engine._pool.put_nowait(None)
        original = db.aiosqlite.connect
        refusals = POOL_SIZE + 1

        def refusing(*args: object, **kwargs: object) -> object:
            nonlocal refusals
            if refusals:
                refusals -= 1
                raise sqlite3.OperationalError("unable to open database file")
            return original(*args, **kwargs)

        monkeypatch.setattr(db.aiosqlite, "connect", refusing)
        for _ in range(POOL_SIZE + 1):
            with pytest.raises(sqlite3.OperationalError):
                await ComputeRow.count()

        async with asyncio.timeout(2):
            assert await ComputeRow.count() == 0


def describe_integer_columns() -> None:
    async def it_reads_every_integer_cell_as_an_int(tmp_path: Path) -> None:
        await connect(tmp_path / "skyward.sqlite")
        await ComputeRow.raw("CREATE TABLE probe (n INTEGER)").run()
        await ComputeRow.raw("INSERT INTO probe (n) VALUES (7), ('1.0'), (2.5)").run()

        rows = await ComputeRow.raw("SELECT n FROM probe ORDER BY rowid").run()

        assert [row["n"] for row in rows] == [7, 1, 2]
        assert all(type(row["n"]) is int for row in rows)

    def it_parses_the_text_one_point_zero_as_one() -> None:
        value = db._integer(b"1.0")

        assert value == 1
        assert type(value) is int


def describe_transaction() -> None:
    async def it_returns_the_same_rows_as_a_pooled_statement(tmp_path: Path) -> None:
        await connect(tmp_path / "skyward.sqlite")
        await ComputeRow.raw("CREATE TABLE probe (n INTEGER, label TEXT)").run()
        await ComputeRow.raw("INSERT INTO probe (n, label) VALUES (1, 'a'), (2, 'b'), (3, NULL)").run()
        query = "SELECT n, label FROM probe WHERE n >= {} ORDER BY n"

        outside = await ComputeRow.raw(query, 2).run()
        async with db.transaction():
            inside = await ComputeRow.raw(query, 2).run()

        assert outside == [{"n": 2, "label": "b"}, {"n": 3, "label": None}]
        assert inside == outside


def describe_indexes() -> None:
    async def it_indexes_tasks_by_compute_and_state(tmp_path: Path) -> None:
        await connect(tmp_path / "skyward.sqlite")

        indexes = await ComputeRow.raw("PRAGMA index_list(tasks)").run()
        columns = await ComputeRow.raw("SELECT name FROM pragma_index_info('tasks_compute_state') ORDER BY seqno").run()

        assert "tasks_compute_state" in {index["name"] for index in indexes}
        assert [column["name"] for column in columns] == ["compute_id", "state"]
