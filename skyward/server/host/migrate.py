"""Schema migration for the server persistence layer."""

import aiosqlite

from skyward.server.host.schema import SCHEMA


async def apply_schema(conn: aiosqlite.Connection) -> None:
    """Apply the persistence schema idempotently and enable WAL.

    Parameters
    ----------
    conn
        An open ``aiosqlite`` connection. The connection is left open; the
        caller retains ownership and is responsible for closing it.
    """
    await conn.execute("PRAGMA journal_mode=WAL")
    for stmt in SCHEMA:
        await conn.execute(stmt)
    await conn.commit()


__all__ = ["apply_schema"]
