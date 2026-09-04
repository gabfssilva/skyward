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
