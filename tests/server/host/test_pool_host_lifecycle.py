"""G1: ``PoolHost`` lifecycle — owns Store + Session across the context."""
from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


@pytest.mark.asyncio
async def test_pool_host_opens_and_closes_cleanly(tmp_path: Path) -> None:
    from skyward.server.host.blobs import Blobs
    from skyward.server.host.pool_host import PoolHost
    from skyward.server.host.store import Store

    store = Store(str(tmp_path / "state.db"))
    await store.open()
    blobs = Blobs(store=store, root=tmp_path / "blobs")

    async with PoolHost(store, blobs, tmp_path / "logs") as host:
        assert host.session is not None
        assert host.store is store

    assert store._write is None
    assert store._read is None


@pytest.mark.asyncio
async def test_subscribe_unsubscribe_round_trip(tmp_path: Path) -> None:
    from skyward.server.host.blobs import Blobs
    from skyward.server.host.pool_host import PoolHost
    from skyward.server.host.store import Store

    store = Store(str(tmp_path / "state.db"))
    await store.open()
    blobs = Blobs(store=store, root=tmp_path / "blobs")

    async with PoolHost(store, blobs, tmp_path / "logs") as host:
        q = host.subscribe("compute:foo")
        assert "compute:foo" in host.subscribers
        assert q in host.subscribers["compute:foo"]

        host.unsubscribe("compute:foo", q)
        assert "compute:foo" not in host.subscribers


@pytest.mark.asyncio
async def test_pool_host_without_enter_raises(tmp_path: Path) -> None:
    from skyward.server.host.blobs import Blobs
    from skyward.server.host.pool_host import PoolHost
    from skyward.server.host.store import Store

    store = Store(str(tmp_path / "state.db"))
    await store.open()
    blobs = Blobs(store=store, root=tmp_path / "blobs")
    host = PoolHost(store, blobs, tmp_path / "logs")
    try:
        with pytest.raises(RuntimeError):
            _ = host.session
    finally:
        await store.close()
