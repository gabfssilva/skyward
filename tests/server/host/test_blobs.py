"""Tests for the ``Blobs`` service."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from skyward.api.spec import Image, Nodes, Spec
from skyward.providers.container.config import Container
from skyward.server.host.blobs import BlobEvicted, Blobs
from skyward.server.host.domain import (
    Compute,
    ComputeSpec,
    PendingRes,
    Provisioning,
    Queued,
    Run,
    SucceededExec,
    Task,
    TaskExecution,
    TaskResult,
)
from skyward.server.host.store import Store


def _t(sec: int = 0) -> datetime:
    return datetime(2026, 4, 14, 12, sec // 60, sec % 60, tzinfo=UTC)


def _compute_spec() -> ComputeSpec:
    return ComputeSpec(
        specs=(Spec(provider=Container(), image=Image(metrics=None)),),
        selection="cheapest",
        nodes=Nodes(desired=1),
        allocation="spot-if-available",
        ttl=timedelta(hours=1),
    )


async def _open_store(tmp_path: Path) -> Store:
    store = Store(str(tmp_path / "test.db"))
    await store.open()
    return store


async def _seed_compute(store: Store, name: str = "c1") -> None:
    await store.put_compute(
        Compute(
            name=name,
            spec=_compute_spec(),
            created_at=_t(0),
            status=Provisioning(started_at=_t(0)),
        )
    )
    await store.put_task(Task(module="m", qualname="f"))


@pytest.mark.asyncio
async def test_put_writes_file_atomically_and_row(tmp_path: Path) -> None:
    store = await _open_store(tmp_path)
    try:
        blobs = Blobs(store=store, root=tmp_path / "blobs")
        bid = await blobs.put(b"hello", kind="payload")
        assert isinstance(bid, int)
        row = await store.get_blob(bid)
        assert row is not None
        assert row.kind == "payload"
        assert row.size == 5
        assert Path(row.path).exists()
        assert Path(row.path).read_bytes() == b"hello"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_path_sharding(tmp_path: Path) -> None:
    root = tmp_path / "blobs"
    store = await _open_store(tmp_path)
    try:
        blobs = Blobs(store=store, root=root)
        bid = await blobs.put(b"abc", kind="payload")
        hex_id = f"{bid:08x}"
        expected = root / hex_id[:2] / f"{hex_id}.bin"
        row = await store.get_blob(bid)
        assert row is not None
        assert Path(row.path) == expected
        assert expected.exists()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_read_returns_bytes(tmp_path: Path) -> None:
    store = await _open_store(tmp_path)
    try:
        blobs = Blobs(store=store, root=tmp_path / "blobs")
        data = b"the quick brown fox"
        bid = await blobs.put(data, kind="result")
        assert await blobs.read(bid) == data
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_read_raises_blob_evicted(tmp_path: Path) -> None:
    store = await _open_store(tmp_path)
    try:
        blobs = Blobs(store=store, root=tmp_path / "blobs")
        bid = await blobs.put(b"x", kind="payload")
        await blobs.evict(bid)
        with pytest.raises(BlobEvicted):
            await blobs.read(bid)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_read_missing_raises_file_not_found(tmp_path: Path) -> None:
    store = await _open_store(tmp_path)
    try:
        blobs = Blobs(store=store, root=tmp_path / "blobs")
        with pytest.raises(FileNotFoundError):
            await blobs.read(9999)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_evict_unlinks_file_and_sets_evicted_at(tmp_path: Path) -> None:
    store = await _open_store(tmp_path)
    try:
        blobs = Blobs(store=store, root=tmp_path / "blobs")
        bid = await blobs.put(b"y", kind="payload")
        row_before = await store.get_blob(bid)
        assert row_before is not None
        path = Path(row_before.path)
        assert path.exists()
        await blobs.evict(bid)
        assert not path.exists()
        row_after = await store.get_blob(bid)
        assert row_after is not None
        assert row_after.evicted_at is not None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_evict_twice_is_idempotent(tmp_path: Path) -> None:
    store = await _open_store(tmp_path)
    try:
        blobs = Blobs(store=store, root=tmp_path / "blobs")
        bid = await blobs.put(b"z", kind="payload")
        await blobs.evict(bid)
        await blobs.evict(bid)
        row = await store.get_blob(bid)
        assert row is not None
        assert row.evicted_at is not None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_evict_missing_is_noop(tmp_path: Path) -> None:
    store = await _open_store(tmp_path)
    try:
        blobs = Blobs(store=store, root=tmp_path / "blobs")
        await blobs.evict(9999)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_gc_skips_live_parents(tmp_path: Path) -> None:
    store = await _open_store(tmp_path)
    try:
        await _seed_compute(store)
        blobs = Blobs(store=store, root=tmp_path / "blobs")
        payload_bid = await blobs.put(b"p", kind="payload")
        result_bid = await blobs.put(b"r", kind="result")
        await store.put_execution(
            TaskExecution(
                id="e1",
                task=("m", "f"),
                compute="c1",
                kind=Run(),
                payload=payload_bid,
                timeout=None,
                client=None,
                submitted_at=_t(0),
                status=Queued(),
            )
        )
        await store.put_result(
            TaskResult(id=0, execution="e1", shard=0, status=PendingRes())
        )
        _ = result_bid
        evicted = await blobs.gc(ttl=timedelta(seconds=0))
        assert evicted == 0
        payload_row = await store.get_blob(payload_bid)
        assert payload_row is not None
        assert payload_row.evicted_at is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_gc_evicts_terminal_parents_past_ttl(tmp_path: Path) -> None:
    store = await _open_store(tmp_path)
    try:
        await _seed_compute(store)
        blobs = Blobs(store=store, root=tmp_path / "blobs")
        payload_bid = await blobs.put(b"payload", kind="payload")
        row = await store.get_blob(payload_bid)
        assert row is not None
        payload_path = Path(row.path)
        assert payload_path.exists()
        past = datetime(2020, 1, 1, tzinfo=UTC)
        await store.put_execution(
            TaskExecution(
                id="e1",
                task=("m", "f"),
                compute="c1",
                kind=Run(),
                payload=payload_bid,
                timeout=None,
                client=None,
                submitted_at=_t(0),
                status=SucceededExec(finished_at=past),
            )
        )
        evicted = await blobs.gc(ttl=timedelta(seconds=60))
        assert evicted == 1
        row_after = await store.get_blob(payload_bid)
        assert row_after is not None
        assert row_after.evicted_at is not None
        assert not payload_path.exists()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_atomicity_on_db_failure(tmp_path: Path) -> None:
    store = await _open_store(tmp_path)
    try:
        root = tmp_path / "blobs"
        blobs = Blobs(store=store, root=root)

        class _BoomTx:
            async def __aenter__(self) -> Any:
                raise RuntimeError("boom")

            async def __aexit__(self, *args: Any) -> Any:
                return None

        def broken_tx() -> Any:
            return _BoomTx()

        original_tx = store.tx
        store.tx = broken_tx  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="boom"):
            await blobs.put(b"nope", kind="payload")
        store.tx = original_tx  # type: ignore[method-assign]

        assert not list(root.rglob("*.tmp"))
        assert not list(root.rglob("*.bin"))
    finally:
        await store.close()
