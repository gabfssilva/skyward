import asyncio
import sqlite3
from pathlib import Path
from typing import Any

import aiosqlite
from piccolo.engine.sqlite import SQLiteEngine, dict_factory
from piccolo.table import Table

from skyward.server.persistence.tables import TABLES

DEFAULT_PATH = Path.home() / ".skyward" / "skyward.sqlite"

PRAGMAS = ("PRAGMA journal_mode=WAL",)
"""Only what sticks to the file. Everything per-connection is applied where the
connections are actually made — on each pooled connection as it opens."""

BUSY_SECONDS = 30
"""How long any statement waits for a write lock before giving up.

The number is the longest write transaction the store ever holds — an offers
refresh rewriting a provider's catalog — with room to spare; past it, the caller
gets ``database is locked``, and a task submit that waited half a minute already
has a bigger problem than the wait.
"""

POOL_SIZE = 8
"""Connections kept, total. WAL admits one writer at a time, so more connections
buy read concurrency only, and eight of those outrun the loop that feeds them."""

MENDS = (
    "UPDATE computes SET spec = json_remove(json_set(spec, '$.nodes.initial', "
    "json_extract(spec, '$.nodes.desired')), '$.nodes.desired') "
    "WHERE json_extract(spec, '$.nodes.desired') IS NOT NULL",
    "UPDATE generations SET spec = json_remove(json_set(spec, '$.nodes.initial', "
    "json_extract(spec, '$.nodes.desired')), '$.nodes.desired') "
    "WHERE json_extract(spec, '$.nodes.desired') IS NOT NULL",
)
"""Rewrites for rows written under an older vocabulary.

``_widen`` adds the columns a release added; this rewrites the JSON already inside
them. A spec written before ``NodeBounds.desired`` became ``initial`` fails to
decode, and the reconciler decodes every compute on every tick — one old row and
the whole plane stalls on it. Each statement matches only rows still carrying the
old shape, so running the list on every start is a no-op after the first.
"""


class PooledSQLiteEngine(SQLiteEngine):
    """Piccolo's SQLite engine, reusing a fixed pool instead of a connection per query.

    The stock engine opens and closes an aiosqlite connection — a thread and up
    to three file descriptors — for every statement. Under a joblib-shaped load
    that walks straight through macOS's default limit of 256 descriptors, and
    SQLite reports the wall as ``unable to open database file``.

    Transactions are untouched: they check a dedicated connection out of
    :meth:`get_connection` and close it, exactly as piccolo wrote it, because a
    transaction's connection carries state the pool must never see.
    """

    def __init__(self, path: str, **connection_kwargs: Any) -> None:
        super().__init__(path=path, **connection_kwargs)
        self._pool: asyncio.Queue[aiosqlite.Connection | None] = asyncio.Queue()
        for _ in range(POOL_SIZE):
            self._pool.put_nowait(None)

    def dispose(self) -> None:
        """Stop every idle connection, without a loop.

        Called when a new engine takes over the tables — synchronous because the
        loop the pool lived on may already be gone.
        """
        while True:
            try:
                connection = self._pool.get_nowait()
            except asyncio.QueueEmpty:
                return
            if connection is not None:
                connection.stop()

    async def _acquire(self) -> aiosqlite.Connection:
        if (connection := await self._pool.get()) is not None:
            return connection

        fresh = aiosqlite.connect(**self.connection_kwargs)
        fresh._thread.daemon = True  # a pooled connection lives until exit, and must not hold it up
        await fresh
        fresh.row_factory = dict_factory  # pyright: ignore[reportAttributeAccessIssue]
        await fresh.execute("PRAGMA foreign_keys = 1")
        await fresh.execute("PRAGMA synchronous = NORMAL")
        return fresh

    async def _run_in_new_connection(
        self,
        query: str,
        args: list[Any] | None = None,
        query_type: str = "generic",
        table: type[Table] | None = None,
    ) -> Any:
        connection = await self._acquire()
        try:
            async with connection.execute(query, args or []) as cursor:
                await connection.commit()
                if query_type == "insert" and self.get_version_sync() < 3.35:
                    assert table is not None
                    pk = await self._get_inserted_pk(cursor, table)
                    result: Any = [{table._meta.primary_key._meta.db_column_name: pk}]
                else:
                    result = await cursor.fetchall()
        except sqlite3.Error:
            self._pool.put_nowait(connection)  # the statement failed; the connection did not
            raise
        except BaseException:
            connection.stop()  # cancelled or broken mid-flight — not a connection to hand out again
            self._pool.put_nowait(None)
            raise
        self._pool.put_nowait(connection)
        return result


_current: PooledSQLiteEngine | None = None
"""The engine the tables are bound to, so the next ``connect`` can retire its pool."""


async def connect(path: Path | None = None) -> SQLiteEngine:
    """Open the store and make it usable by more than one process.

    WAL is what lets a `sky.Compute` in one script and the daemon in another
    write to the same database without an exclusive lock file. It is a journal
    mode, not a change feed — nothing reads the WAL itself.
    """
    target = path or DEFAULT_PATH
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    global _current
    if _current is not None:
        _current.dispose()

    engine = _current = PooledSQLiteEngine(path=str(target), timeout=BUSY_SECONDS)
    for table in TABLES:
        table._meta.db = engine

    for pragma in PRAGMAS:
        await TABLES[0].raw(pragma).run()

    for table in TABLES:
        await table.create_table(if_not_exists=True).run()
        await _widen(table)

    for statement in MENDS:
        await TABLES[0].raw(statement).run()

    return engine


async def _widen(table: type[Table]) -> None:
    """Add the columns the model has and the file does not.

    ``if_not_exists`` is the right thing on every start after the first and no help
    at all when a release adds a column: the file keeps the shape it was created
    with, and the first read of the new model fails on a column that is not there.
    Adding the missing ones is the whole of what a migration would do here, because
    a column added to a table that already has rows can only ever be nullable — a
    row written before the column exists has nothing to say about it.
    """
    present = {column["name"] for column in await table.raw(f"PRAGMA table_info({table._meta.tablename})").run()}
    for column in table._meta.columns:
        if column._meta.db_column_name not in present:
            await table.raw(f"ALTER TABLE {table._meta.tablename} ADD COLUMN {column.ddl}").run()
