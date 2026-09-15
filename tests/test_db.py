"""The store's connection pool, under the failures it exists to absorb."""

import asyncio
import sqlite3
from pathlib import Path

import pytest

from skyward.server.persistence import db
from skyward.server.persistence.db import POOL_SIZE, connect
from skyward.server.persistence.events import EventStore
from skyward.server.persistence.tables import ComputeRow, EventRow, ExecutionRow
from skyward.shared.events import ConsoleEvent

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


def describe_a_log_written_before_the_node_column() -> None:
    async def an_event_is_given_the_node_its_own_payload_names(tmp_path: Path) -> None:
        """A file from before the column, opened by a daemon that has it: the rows keep their node."""
        path = tmp_path / "skyward.sqlite"
        await connect(path)
        await EventRow.raw("DROP INDEX events_node_id").run()
        await EventRow.raw("ALTER TABLE events DROP COLUMN node_id").run()
        await EventRow.raw(
            "INSERT INTO events (type, compute_id, payload, created_at) VALUES ('node.console', 'cmp_a', {}, '2026-01-01 00:00:00+00:00')",
            '{"type":"node.console","compute":"cmp_a","node":"nod_7","content":"printed"}',
        ).run()

        await connect(path)

        assert [row["node_id"] for row in await EventRow.select(EventRow.node_id)] == ["nod_7"]
        assert await EventRow.count().where(EventRow.node_id == "nod_7") == 1, "asked for through the index the added column carries"
        assert [row["integrity_check"] for row in await EventRow.raw("PRAGMA integrity_check").run()] == ["ok"]


def describe_a_log_whose_lines_named_the_execution_as_their_task() -> None:
    async def a_line_is_given_the_task_and_keeps_the_execution_under_its_own_name(tmp_path: Path) -> None:
        path = tmp_path / "skyward.sqlite"
        await connect(path)
        await ExecutionRow(id="exe_1", task_id="tsk_1", ordinal=1, state="succeeded").save().run()
        await EventRow.raw(
            "INSERT INTO events (type, compute_id, node_id, task_id, payload, created_at) "
            "VALUES ('node.console', 'cmp_a', 'nod_7', 'exe_1', {}, '2026-01-01 00:00:00+00:00')",
            '{"type":"node.console","compute":"cmp_a","node":"nod_7","content":"printed","task":"exe_1"}',
        ).run()

        await connect(path)

        (entry,) = (await EventStore().log(None, 10, task="tsk_1")).items
        assert entry.data ==ConsoleEvent(compute="cmp_a", node="nod_7", content="printed", task="tsk_1", execution="exe_1")
