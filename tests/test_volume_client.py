"""Tests for VolumeClient, ObjectStore protocol, and S3ObjectStore."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from skyward.api.spec import Volume
from skyward.api.volume_client import VolumeClient
from skyward.infra.object_store import S3ObjectStore
from skyward.providers.provider import ObjectStore

# =============================================================================
# ObjectStore protocol
# =============================================================================


class TestObjectStoreProtocol:
    def test_s3_object_store_satisfies_protocol(self):
        assert hasattr(S3ObjectStore, "upload_file")
        assert hasattr(S3ObjectStore, "download_file")
        assert hasattr(S3ObjectStore, "list_objects")
        assert hasattr(S3ObjectStore, "delete_objects")
        assert hasattr(S3ObjectStore, "head_object")

    def test_object_store_is_runtime_checkable(self):
        mock_s3 = MagicMock()
        store = S3ObjectStore(mock_s3)
        assert isinstance(store, ObjectStore)


# =============================================================================
# S3ObjectStore
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
            Bucket="bucket", Key="data.csv", Body=b"a,b\n1,2", ContentLength=7,
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
# VolumeClient
# =============================================================================


def _make_mock_provider(store: ObjectStore | None = None) -> Any:
    """Create a mock Mountable provider with an object_store context manager."""
    from collections.abc import AsyncIterator
    from contextlib import asynccontextmanager

    mock_store = store or MagicMock(spec=ObjectStore)
    provider = MagicMock()
    provider.mount_endpoint = AsyncMock()

    @asynccontextmanager
    async def _object_store() -> AsyncIterator[ObjectStore]:
        yield mock_store  # type: ignore[misc]

    provider.object_store = _object_store
    return provider, mock_store


class TestVolumeClientLifecycle:
    def test_enter_exit(self):
        vol = Volume(bucket="b", mount="/data")
        provider, _store = _make_mock_provider()

        config = MagicMock()
        config.type = "test"
        config.create_provider = AsyncMock(return_value=provider)

        with VolumeClient(vol, provider=config) as vc:
            assert vc._store is not None

        assert vc._loop is None

    def test_non_mountable_raises(self):
        vol = Volume(bucket="b", mount="/data")
        non_mountable_provider = MagicMock(spec=[])

        config = MagicMock()
        config.type = "test"
        config.create_provider = AsyncMock(return_value=non_mountable_provider)

        with pytest.raises(TypeError, match="does not support volumes"), VolumeClient(vol, provider=config):
            pass


class TestVolumeClientResolveKey:
    def test_with_prefix(self):
        vc = VolumeClient.__new__(VolumeClient)
        vc._volume = Volume(bucket="b", mount="/data", prefix="train/")
        assert vc._resolve_key("model.pt") == "train/model.pt"

    def test_without_prefix(self):
        vc = VolumeClient.__new__(VolumeClient)
        vc._volume = Volume(bucket="b", mount="/data")
        assert vc._resolve_key("model.pt") == "model.pt"


class TestVolumeClientUpload:
    @pytest.mark.asyncio()
    async def test_upload_file(self, tmp_path: Path):
        store = AsyncMock(spec=ObjectStore)
        vc = VolumeClient.__new__(VolumeClient)
        vc._volume = Volume(bucket="b", mount="/data", prefix="p/")
        vc._store = store

        f = tmp_path / "model.pt"
        f.write_text("weights")

        await vc._upload(f, None)
        store.upload_file.assert_awaited_once_with("b", "p/model.pt", f)

    @pytest.mark.asyncio()
    async def test_upload_file_with_key(self, tmp_path: Path):
        store = AsyncMock(spec=ObjectStore)
        vc = VolumeClient.__new__(VolumeClient)
        vc._volume = Volume(bucket="b", mount="/data", prefix="p/")
        vc._store = store

        f = tmp_path / "model.pt"
        f.write_text("weights")

        await vc._upload(f, "v2/model.pt")
        store.upload_file.assert_awaited_once_with("b", "p/v2/model.pt", f)

    @pytest.mark.asyncio()
    async def test_upload_directory(self, tmp_path: Path):
        store = AsyncMock(spec=ObjectStore)
        vc = VolumeClient.__new__(VolumeClient)
        vc._volume = Volume(bucket="b", mount="/data", prefix="p/")
        vc._store = store

        d = tmp_path / "data"
        d.mkdir()
        (d / "a.csv").write_text("1")
        (d / "sub").mkdir()
        (d / "sub" / "b.csv").write_text("2")

        await vc._upload(d, None)
        assert store.upload_file.await_count == 2
        uploaded_keys = sorted(call.args[1] for call in store.upload_file.await_args_list)
        assert uploaded_keys == ["p/data/a.csv", "p/data/sub/b.csv"]


class TestVolumeClientDownload:
    @pytest.mark.asyncio()
    async def test_download_single_file(self, tmp_path: Path):
        store = AsyncMock(spec=ObjectStore)
        store.head_object.return_value = True
        vc = VolumeClient.__new__(VolumeClient)
        vc._volume = Volume(bucket="b", mount="/data", prefix="p/")
        vc._store = store

        dest = tmp_path / "model.pt"
        await vc._download("model.pt", dest)
        store.download_file.assert_awaited_once_with("b", "p/model.pt", dest)

    @pytest.mark.asyncio()
    async def test_download_prefix(self, tmp_path: Path):
        store = AsyncMock(spec=ObjectStore)
        store.head_object.return_value = False
        store.list_objects.return_value = ["p/ckpts/a.pt", "p/ckpts/b.pt"]
        vc = VolumeClient.__new__(VolumeClient)
        vc._volume = Volume(bucket="b", mount="/data", prefix="p/")
        vc._store = store

        dest = tmp_path / "local"
        await vc._download("ckpts/", dest)
        assert store.download_file.await_count == 2

    @pytest.mark.asyncio()
    async def test_download_not_found(self):
        store = AsyncMock(spec=ObjectStore)
        store.head_object.return_value = False
        store.list_objects.return_value = []
        vc = VolumeClient.__new__(VolumeClient)
        vc._volume = Volume(bucket="b", mount="/data")
        vc._store = store

        with pytest.raises(FileNotFoundError):
            await vc._download("missing.pt", Path("/tmp/out"))


class TestVolumeClientLs:
    @pytest.mark.asyncio()
    async def test_ls_strips_prefix(self):
        store = AsyncMock(spec=ObjectStore)
        store.list_objects.return_value = ["train/a.csv", "train/b.csv"]
        vc = VolumeClient.__new__(VolumeClient)
        vc._volume = Volume(bucket="b", mount="/data", prefix="train/")
        vc._store = store

        result = await vc._ls("")
        assert result == ["a.csv", "b.csv"]
        store.list_objects.assert_awaited_once_with("b", "train/")

    @pytest.mark.asyncio()
    async def test_ls_with_subprefix(self):
        store = AsyncMock(spec=ObjectStore)
        store.list_objects.return_value = ["train/sub/x.csv"]
        vc = VolumeClient.__new__(VolumeClient)
        vc._volume = Volume(bucket="b", mount="/data", prefix="train/")
        vc._store = store

        result = await vc._ls("sub/")
        assert result == ["sub/x.csv"]


class TestVolumeClientExists:
    @pytest.mark.asyncio()
    async def test_exists_true(self):
        store = AsyncMock(spec=ObjectStore)
        store.head_object.return_value = True
        vc = VolumeClient.__new__(VolumeClient)
        vc._volume = Volume(bucket="b", mount="/data", prefix="p/")
        vc._store = store

        assert await vc._exists("model.pt") is True
        store.head_object.assert_awaited_once_with("b", "p/model.pt")

    @pytest.mark.asyncio()
    async def test_exists_false(self):
        store = AsyncMock(spec=ObjectStore)
        store.head_object.return_value = False
        vc = VolumeClient.__new__(VolumeClient)
        vc._volume = Volume(bucket="b", mount="/data")
        vc._store = store

        assert await vc._exists("missing.pt") is False


class TestVolumeClientRm:
    @pytest.mark.asyncio()
    async def test_rm_single_object(self):
        store = AsyncMock(spec=ObjectStore)
        store.head_object.return_value = True
        vc = VolumeClient.__new__(VolumeClient)
        vc._volume = Volume(bucket="b", mount="/data", prefix="p/")
        vc._store = store

        await vc._rm("old.pt")
        store.delete_objects.assert_awaited_once_with("b", ["p/old.pt"])

    @pytest.mark.asyncio()
    async def test_rm_prefix(self):
        store = AsyncMock(spec=ObjectStore)
        store.head_object.return_value = False
        store.list_objects.return_value = ["p/old/a.pt", "p/old/b.pt"]
        vc = VolumeClient.__new__(VolumeClient)
        vc._volume = Volume(bucket="b", mount="/data", prefix="p/")
        vc._store = store

        await vc._rm("old/")
        store.delete_objects.assert_awaited_once_with("b", ["p/old/a.pt", "p/old/b.pt"])

    @pytest.mark.asyncio()
    async def test_rm_nothing_found(self):
        store = AsyncMock(spec=ObjectStore)
        store.head_object.return_value = False
        store.list_objects.return_value = []
        vc = VolumeClient.__new__(VolumeClient)
        vc._volume = Volume(bucket="b", mount="/data")
        vc._store = store

        await vc._rm("ghost.pt")
        store.delete_objects.assert_not_awaited()


# =============================================================================
# Provider object_store() method existence
# =============================================================================


class TestProviderObjectStore:
    def test_aws_has_object_store(self):
        from skyward.providers.aws.provider import AWSProvider

        assert hasattr(AWSProvider, "object_store")

    def test_gcp_has_object_store(self):
        from skyward.providers.gcp.provider import GCPProvider

        assert hasattr(GCPProvider, "object_store")

    def test_hyperstack_has_object_store(self):
        from skyward.providers.hyperstack.provider import HyperstackProvider

        assert hasattr(HyperstackProvider, "object_store")

    def test_runpod_has_object_store(self):
        from skyward.providers.runpod.provider import RunPodProvider

        assert hasattr(RunPodProvider, "object_store")


# =============================================================================
# Exports
# =============================================================================


class TestVolumeClientExports:
    def test_importable_from_skyward(self):
        import skyward as sky

        assert hasattr(sky, "VolumeClient")

    def test_in_all(self):
        import skyward

        assert "VolumeClient" in skyward.__all__

    def test_s3_object_store_importable_from_infra(self):
        from skyward.infra import S3ObjectStore

        assert S3ObjectStore is not None
