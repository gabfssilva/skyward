"""Tests for blob and error repository methods on ``Store``."""

from __future__ import annotations

from pathlib import Path

import pytest

from skyward.server.host.store import Store


async def _open(tmp_path: Path) -> Store:
    store = Store(str(tmp_path / "test.db"))
    await store.open()
    return store


@pytest.mark.asyncio
async def test_put_blob_returns_id_monotonic(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        a = await store.put_blob(path="/tmp/a", size=1, kind="payload")
        b = await store.put_blob(path="/tmp/b", size=2, kind="payload")
        c = await store.put_blob(path="/tmp/c", size=3, kind="result")
        assert a < b < c
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_get_blob_roundtrip(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        bid = await store.put_blob(
            path="/tmp/x.bin", size=128, kind="payload", sha256="abc123"
        )
        got = await store.get_blob(bid)
        assert got is not None
        assert got.id == bid
        assert got.path == Path("/tmp/x.bin")
        assert got.size == 128
        assert got.kind == "payload"
        assert got.sha256 == "abc123"
        assert got.evicted_at is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_evict_blob_sets_evicted_at(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        bid = await store.put_blob(path="/tmp/e", size=1, kind="result")
        before = await store.get_blob(bid)
        assert before is not None
        assert before.evicted_at is None
        await store.evict_blob(bid)
        after = await store.get_blob(bid)
        assert after is not None
        assert after.evicted_at is not None
        assert after.path == Path("/tmp/e")
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_get_blob_missing_returns_none(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        assert await store.get_blob(9999) is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_put_error_returns_id_monotonic(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        a = await store.put_error(type="RuntimeError", message="a")
        b = await store.put_error(type="ValueError", message="b")
        assert a < b
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_put_error_persists_traceback(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        eid = await store.put_error(
            type="RuntimeError", message="boom", traceback="line1\nline2"
        )
        assert eid > 0
    finally:
        await store.close()
