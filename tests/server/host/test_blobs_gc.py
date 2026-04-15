"""I4: ``Blobs.gc_orphans`` reconciles file + row drift on startup."""
from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


@pytest.mark.asyncio
async def test_gc_orphans_deletes_file_without_row(tmp_path: Path) -> None:
    from skyward.server.host.blobs import Blobs
    from skyward.server.host.store import Store

    store = Store(str(tmp_path / "db"))
    await store.open()
    try:
        root = tmp_path / "blobs"
        shard = root / "12"
        shard.mkdir(parents=True)
        stray = shard / "12345678.bin"
        stray.write_bytes(b"stray")

        blobs = Blobs(store=store, root=root)
        removed = await blobs.gc_orphans()
        assert removed == 1
        assert not stray.exists()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_gc_orphans_evicts_row_without_file(tmp_path: Path) -> None:
    from skyward.server.host.blobs import Blobs
    from skyward.server.host.store import Store

    store = Store(str(tmp_path / "db"))
    await store.open()
    try:
        root = tmp_path / "blobs"
        blobs = Blobs(store=store, root=root)
        blob_id = await blobs.put(b"abc", kind="payload")

        row = await store.get_blob(blob_id)
        assert row is not None
        Path(row.path).unlink()

        removed = await blobs.gc_orphans()
        assert removed == 1
        row_after = await store.get_blob(blob_id)
        assert row_after is not None
        assert row_after.evicted_at is not None
    finally:
        await store.close()
