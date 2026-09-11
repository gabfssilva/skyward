"""Blobs kept as content-defined chunks, and the files from before chunking."""

import hashlib
import os
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from litestar.testing import AsyncTestClient

from skyward.server.http.app import create_app, services
from skyward.server.persistence.db import connect
from skyward.server.persistence.functions import BlobStore, FunctionStore, _split
from skyward.server.persistence.tables import BlobRow, ChunkRow

pytestmark = pytest.mark.local

MIB = 1024 * 1024


def sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


async def tables() -> set[str]:
    rows = await BlobRow.raw("SELECT name FROM sqlite_master WHERE type = 'table'").run()
    return {row["name"] for row in rows}


@pytest.fixture
async def store(tmp_path: Path) -> BlobStore:
    await connect(tmp_path / "skyward.sqlite")
    return BlobStore()


def describe_a_blob() -> None:
    @pytest.mark.parametrize("size", [0, 17, 5 * MIB])
    async def it_reads_back_byte_identical(store: BlobStore, size: int) -> None:
        blob = os.urandom(size)
        assert await store.put(sha(blob), blob) is True
        assert await BlobStore().get(sha(blob)) == blob

    async def it_splits_several_mebibytes_into_many_chunks(store: BlobStore) -> None:
        blob = os.urandom(5 * MIB)
        await store.put(sha(blob), blob)
        assert await ChunkRow.count() > 16
        assert await BlobStore().get(sha(blob)) == blob


def describe_shared_content() -> None:
    async def it_stores_a_shared_region_once(store: BlobStore) -> None:
        region = os.urandom(4 * MIB)
        first = os.urandom(300_000) + region
        second = os.urandom(200_000) + region + os.urandom(50_000)

        await store.put(sha(first), first)
        after_first = await ChunkRow.count()
        await store.put(sha(second), second)
        added = await ChunkRow.count() - after_first

        assert added < len(set(_split(second))) // 2
        reader = BlobStore()
        assert await reader.get(sha(first)) == first
        assert await reader.get(sha(second)) == second

    async def putting_stored_content_returns_false_and_writes_nothing(store: BlobStore) -> None:
        blob = os.urandom(MIB)
        assert await store.put(sha(blob), blob) is True
        chunks, blobs = await ChunkRow.count(), await BlobRow.count()

        assert await store.put(sha(blob), blob) is False
        assert (await ChunkRow.count(), await BlobRow.count()) == (chunks, blobs)


def describe_a_database_from_before_chunking() -> None:
    async def its_whole_blobs_are_readable_before_and_after_rechunking(tmp_path: Path) -> None:
        path = tmp_path / "skyward.sqlite"
        legacy = [b"", b"small", os.urandom(3 * MIB)]
        with sqlite3.connect(path) as old:
            old.execute("CREATE TABLE blobs (sha256 VARCHAR PRIMARY KEY, data BLOB, created_at TIMESTAMPTZ)")
            old.executemany(
                "INSERT INTO blobs VALUES (?, ?, ?)",
                [(sha(blob), blob, "2025-01-01 00:00:00+00:00") for blob in legacy],
            )
        old.close()

        await connect(path)
        assert {"blobs_legacy", "blobs", "chunks"} <= await tables()

        store = BlobStore()
        for blob in legacy:
            assert await store.exists(sha(blob))
            assert await store.get(sha(blob)) == blob

        await store.rechunk()

        assert "blobs_legacy" not in await tables()
        reader = BlobStore()
        for blob in legacy:
            assert await reader.exists(sha(blob))
            assert await reader.get(sha(blob)) == blob


@pytest.fixture
async def http(tmp_path: Path) -> AsyncIterator[AsyncTestClient]:
    await connect(tmp_path / "skyward.sqlite")
    async with AsyncTestClient(app=create_app(services(), logging=False)) as client:
        yield client


def describe_checking_existence() -> None:
    async def a_missing_blob_answers_404(http: AsyncTestClient) -> None:
        response = await http.head(f"/v1/blobs/{sha(b'nowhere')}")
        assert response.status_code == 404

    async def a_stored_blob_answers_200(http: AsyncTestClient) -> None:
        blob = os.urandom(4096)
        uploaded = await http.put(f"/v1/blobs/{sha(blob)}", content=blob)
        assert uploaded.status_code == 201

        response = await http.head(f"/v1/blobs/{sha(blob)}")
        assert response.status_code == 200


def describe_registering_a_function() -> None:
    async def a_registered_function_is_not_stored_again_even_with_a_body_that_does_not_match(store: BlobStore) -> None:
        functions = FunctionStore(store)
        body = os.urandom(10_000)
        registered, created = await functions.register(sha(body), body, "train")
        assert created is True
        chunks, blobs = await ChunkRow.count(), await BlobRow.count()

        again, created_again = await functions.register(sha(body), b"not the body", "train")

        assert (again, created_again) == (registered, False)
        assert (await ChunkRow.count(), await BlobRow.count()) == (chunks, blobs)
        assert await store.get(sha(body)) == body
