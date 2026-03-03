"""Tests for S3ObjectStore and Storage CRUD internals."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from skyward.infra.object_store import S3ObjectStore

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


# =============================================================================
# S3ObjectStore (internal implementation, still used by Storage)
# =============================================================================


class TestS3ObjectStore:
    @pytest.fixture()
    def s3(self):
        return AsyncMock()

    @pytest.fixture()
    def store(self, s3: AsyncMock):
        return S3ObjectStore(s3)

    @pytest.mark.asyncio()
    async def test_upload_file(self, store: S3ObjectStore, s3: AsyncMock, tmp_path: Path):
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2")
        await store.upload_file("bucket", "data.csv", f)
        s3.put_object.assert_awaited_once_with(
            Bucket="bucket", Key="data.csv", Body=b"a,b\n1,2",
        )

    @pytest.mark.asyncio()
    async def test_download_file(self, store: S3ObjectStore, s3: AsyncMock, tmp_path: Path):
        dest = tmp_path / "sub" / "model.pt"
        await store.download_file("bucket", "model.pt", dest)
        assert dest.parent.exists()
        s3.download_file.assert_awaited_once_with("bucket", "model.pt", str(dest))

    @pytest.mark.asyncio()
    async def test_list_objects_single_page(self, store: S3ObjectStore, s3: AsyncMock):
        s3.list_objects_v2.return_value = {
            "Contents": [{"Key": "a.txt"}, {"Key": "b.txt"}],
            "IsTruncated": False,
        }
        keys = await store.list_objects("bucket", "prefix/")
        assert keys == ["a.txt", "b.txt"]

    @pytest.mark.asyncio()
    async def test_list_objects_paginated(self, store: S3ObjectStore, s3: AsyncMock):
        s3.list_objects_v2.side_effect = [
            {
                "Contents": [{"Key": "a.txt"}],
                "IsTruncated": True,
                "NextContinuationToken": "tok",
            },
            {
                "Contents": [{"Key": "b.txt"}],
                "IsTruncated": False,
            },
        ]
        keys = await store.list_objects("bucket", "")
        assert keys == ["a.txt", "b.txt"]
        assert s3.list_objects_v2.await_count == 2

    @pytest.mark.asyncio()
    async def test_list_objects_empty(self, store: S3ObjectStore, s3: AsyncMock):
        s3.list_objects_v2.return_value = {"IsTruncated": False}
        keys = await store.list_objects("bucket", "missing/")
        assert keys == []

    @pytest.mark.asyncio()
    async def test_delete_objects_single_batch(self, store: S3ObjectStore, s3: AsyncMock):
        await store.delete_objects("bucket", ["a.txt", "b.txt"])
        s3.delete_objects.assert_awaited_once_with(
            Bucket="bucket",
            Delete={"Objects": [{"Key": "a.txt"}, {"Key": "b.txt"}]},
        )

    @pytest.mark.asyncio()
    async def test_delete_objects_batched(self, store: S3ObjectStore, s3: AsyncMock):
        keys = [f"file_{i}.txt" for i in range(1500)]
        await store.delete_objects("bucket", keys)
        assert s3.delete_objects.await_count == 2

    @pytest.mark.asyncio()
    async def test_head_object_exists(self, store: S3ObjectStore, s3: AsyncMock):
        s3.head_object.return_value = {}
        assert await store.head_object("bucket", "key") is True

    @pytest.mark.asyncio()
    async def test_head_object_not_found(self, store: S3ObjectStore, s3: AsyncMock):
        from botocore.exceptions import ClientError

        s3.head_object.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}},
            "HeadObject",
        )
        assert await store.head_object("bucket", "key") is False

    @pytest.mark.asyncio()
    async def test_head_object_other_error_raises(self, store: S3ObjectStore, s3: AsyncMock):
        from botocore.exceptions import ClientError

        s3.head_object.side_effect = ClientError(
            {"Error": {"Code": "403", "Message": "Forbidden"}},
            "HeadObject",
        )
        with pytest.raises(ClientError):
            await store.head_object("bucket", "key")


# =============================================================================
# Storage CRUD internals
# =============================================================================


class TestStorageUpload:
    @pytest.mark.asyncio()
    async def test_upload_file(self, tmp_path: Path):
        from skyward.storage import _upload

        s3 = AsyncMock()
        f = tmp_path / "model.pt"
        f.write_text("weights")
        await _upload(s3, "b", f, None)
        s3.put_object.assert_awaited_once()

    @pytest.mark.asyncio()
    async def test_upload_file_with_key(self, tmp_path: Path):
        from skyward.storage import _upload

        s3 = AsyncMock()
        f = tmp_path / "model.pt"
        f.write_text("weights")
        await _upload(s3, "b", f, "v2/model.pt")
        s3.put_object.assert_awaited_once()
        call_kwargs = s3.put_object.await_args
        assert call_kwargs.kwargs["Key"] == "v2/model.pt"

    @pytest.mark.asyncio()
    async def test_upload_directory(self, tmp_path: Path):
        from skyward.storage import _upload

        s3 = AsyncMock()
        d = tmp_path / "data"
        d.mkdir()
        (d / "a.csv").write_text("1")
        (d / "sub").mkdir()
        (d / "sub" / "b.csv").write_text("2")
        await _upload(s3, "b", d, None)
        assert s3.put_object.await_count == 2

class TestStorageDownload:
    @pytest.mark.asyncio()
    async def test_download_single_file(self, tmp_path: Path):
        from skyward.storage import _download

        s3 = AsyncMock()
        dest = tmp_path / "model.pt"
        await _download(s3, "b", "model.pt", dest)
        s3.download_file.assert_awaited_once()

    @pytest.mark.asyncio()
    async def test_download_not_found(self):
        from skyward.storage import _download

        s3 = AsyncMock()
        s3.download_file.side_effect = Exception("not found")
        with pytest.raises(Exception):
            await _download(s3, "b", "missing.pt", Path("/tmp/out"))


class TestStorageRm:
    @pytest.mark.asyncio()
    async def test_rm_single_object(self):
        from skyward.storage import _rm

        s3 = AsyncMock()
        s3.head_object.return_value = {}
        await _rm(s3, "b", "old.pt")
        s3.delete_objects.assert_awaited_once()

    @pytest.mark.asyncio()
    async def test_rm_prefix(self):
        from botocore.exceptions import ClientError

        from skyward.storage import _rm

        s3 = AsyncMock()
        s3.head_object.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject",
        )
        s3.list_objects_v2.return_value = {
            "Contents": [{"Key": "old/a.pt"}, {"Key": "old/b.pt"}],
            "IsTruncated": False,
        }
        await _rm(s3, "b", "old/")
        s3.delete_objects.assert_awaited_once()

    @pytest.mark.asyncio()
    async def test_rm_nothing_found(self):
        from botocore.exceptions import ClientError

        from skyward.storage import _rm

        s3 = AsyncMock()
        s3.head_object.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject",
        )
        s3.list_objects_v2.return_value = {"IsTruncated": False}
        await _rm(s3, "b", "ghost.pt")
        s3.delete_objects.assert_not_awaited()


# =============================================================================
# Exports
# =============================================================================


class TestStorageExports:
    def test_s3_object_store_importable_from_infra(self):
        from skyward.infra import S3ObjectStore

        assert S3ObjectStore is not None

    def test_storage_importable(self):
        import skyward as sky

        assert hasattr(sky, "Storage")

    def test_providers_have_storage(self):
        from skyward.providers.aws.provider import AWSProvider
        from skyward.providers.gcp.provider import GCPProvider
        from skyward.providers.hyperstack.provider import HyperstackProvider
        from skyward.providers.runpod.provider import RunPodProvider

        assert hasattr(AWSProvider, "storage")
        assert hasattr(GCPProvider, "storage")
        assert hasattr(HyperstackProvider, "storage")
        assert hasattr(RunPodProvider, "storage")
