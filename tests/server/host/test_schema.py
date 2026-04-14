"""Tests for the persistence schema and migration."""

import aiosqlite
import pytest

from skyward.server.host.migrate import apply_schema

EXPECTED_TABLES = {
    "providers",
    "compute",
    "nodes",
    "blobs",
    "errors",
    "tasks",
    "task_executions",
    "task_results",
    "events",
}

EXPECTED_INDEXES = {
    "idx_providers_type",
    "idx_compute_status",
    "idx_compute_activity",
    "idx_nodes_compute",
    "idx_blobs_live",
    "idx_errors_type",
    "idx_exec_compute_status",
    "idx_exec_task",
    "idx_exec_group",
    "idx_exec_active",
    "idx_results_execution",
    "idx_results_node",
    "idx_events_agg",
}


@pytest.mark.asyncio
async def test_apply_schema_creates_all_tables() -> None:
    async with aiosqlite.connect(":memory:") as conn:
        await apply_schema(conn)
        cursor = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        rows = await cursor.fetchall()
        names = {row[0] for row in rows} - {"sqlite_sequence"}
        assert names == EXPECTED_TABLES


@pytest.mark.asyncio
async def test_apply_schema_creates_all_indexes() -> None:
    async with aiosqlite.connect(":memory:") as conn:
        await apply_schema(conn)
        cursor = await conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='index' AND name NOT LIKE 'sqlite_%'"
        )
        rows = await cursor.fetchall()
        names = {row[0] for row in rows}
        assert names == EXPECTED_INDEXES


@pytest.mark.asyncio
async def test_apply_schema_is_idempotent() -> None:
    async with aiosqlite.connect(":memory:") as conn:
        await apply_schema(conn)
        await apply_schema(conn)
        cursor = await conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table'"
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] >= len(EXPECTED_TABLES)


@pytest.mark.asyncio
async def test_apply_schema_sets_wal(tmp_path) -> None:
    db_path = tmp_path / "test.db"
    async with aiosqlite.connect(db_path) as conn:
        await apply_schema(conn)
        cursor = await conn.execute("PRAGMA journal_mode")
        row = await cursor.fetchone()
        assert row is not None
        assert row[0].lower() == "wal"
