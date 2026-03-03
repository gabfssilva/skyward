from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class TestStorage:
    def test_basic_creation(self):
        from skyward.storage import Storage

        s = Storage(endpoint="https://s3.us-east-1.amazonaws.com")
        assert s.endpoint == "https://s3.us-east-1.amazonaws.com"
        assert s.access_key is None
        assert s.secret_key is None
        assert s.path_style is False

    def test_with_credentials(self):
        from skyward.storage import Storage

        s = Storage(
            endpoint="https://abc.r2.cloudflarestorage.com",
            access_key="AK",
            secret_key="SK",
        )
        assert s.access_key == "AK"
        assert s.secret_key == "SK"

    def test_path_style(self):
        from skyward.storage import Storage

        s = Storage(endpoint="https://obj.nexgencloud.io", path_style=True)
        assert s.path_style is True

    def test_with_callable_credentials(self):
        from skyward.storage import Storage

        s = Storage(
            endpoint="https://s3.amazonaws.com",
            access_key=lambda: "AK_LAZY",
            secret_key=lambda: "SK_LAZY",
        )
        assert callable(s.access_key)
        assert callable(s.secret_key)

    def test_frozen(self):
        from skyward.storage import Storage

        s = Storage(endpoint="https://s3.amazonaws.com")
        with pytest.raises(AttributeError):
            s.endpoint = "other"  # type: ignore[misc]


class TestResolve:
    @pytest.mark.asyncio()
    async def test_resolve_strings_returns_self(self):
        from skyward.storage import Storage

        s = Storage(endpoint="https://s3.amazonaws.com", access_key="AK", secret_key="SK")
        resolved = await s.resolve()
        assert resolved is s

    @pytest.mark.asyncio()
    async def test_resolve_none_returns_self(self):
        from skyward.storage import Storage

        s = Storage(endpoint="https://s3.amazonaws.com")
        resolved = await s.resolve()
        assert resolved is s

    @pytest.mark.asyncio()
    async def test_resolve_sync_callable(self):
        from skyward.storage import Storage

        s = Storage(
            endpoint="https://s3.amazonaws.com",
            access_key=lambda: "AK_RESOLVED",
            secret_key=lambda: "SK_RESOLVED",
        )
        resolved = await s.resolve()
        assert resolved is not s
        assert resolved.access_key == "AK_RESOLVED"
        assert resolved.secret_key == "SK_RESOLVED"
        assert resolved.endpoint == s.endpoint
        assert resolved.path_style == s.path_style

    @pytest.mark.asyncio()
    async def test_resolve_async_callable(self):
        from skyward.storage import Storage

        async def get_ak() -> str:
            return "AK_ASYNC"

        async def get_sk() -> str:
            return "SK_ASYNC"

        s = Storage(
            endpoint="https://s3.amazonaws.com",
            access_key=get_ak,
            secret_key=get_sk,
        )
        resolved = await s.resolve()
        assert resolved.access_key == "AK_ASYNC"
        assert resolved.secret_key == "SK_ASYNC"

    @pytest.mark.asyncio()
    async def test_resolve_mixed(self):
        from skyward.storage import Storage

        async def get_sk() -> str:
            return "SK_ASYNC"

        s = Storage(
            endpoint="https://s3.amazonaws.com",
            access_key="AK_STATIC",
            secret_key=get_sk,
        )
        resolved = await s.resolve()
        assert resolved.access_key == "AK_STATIC"
        assert resolved.secret_key == "SK_ASYNC"

    @pytest.mark.asyncio()
    async def test_resolve_callable_with_preset(self):
        from skyward.storage.presets import R2

        s = R2(
            account_id="abc123",
            access_key=lambda: "AK_LAZY",
            secret_key=lambda: "SK_LAZY",
        )
        resolved = await s.resolve()
        assert resolved.access_key == "AK_LAZY"
        assert resolved.secret_key == "SK_LAZY"
        assert resolved.endpoint == "https://abc123.r2.cloudflarestorage.com"


class TestPresets:
    def test_r2(self):
        from skyward.storage import Storage
        from skyward.storage.presets import R2

        s = R2(account_id="abc123", access_key="AK", secret_key="SK")
        assert isinstance(s, Storage)
        assert s.endpoint == "https://abc123.r2.cloudflarestorage.com"
        assert s.access_key == "AK"

    def test_s3_default_region(self):
        from skyward.storage import Storage
        from skyward.storage.presets import S3

        s = S3()
        assert isinstance(s, Storage)
        assert s.endpoint == "https://s3.us-east-1.amazonaws.com"
        assert s.access_key is None

    def test_s3_with_credentials(self):
        from skyward.storage.presets import S3

        s = S3(region="eu-west-1", access_key="AK", secret_key="SK")
        assert s.endpoint == "https://s3.eu-west-1.amazonaws.com"
        assert s.access_key == "AK"

    def test_gcs(self):
        from skyward.storage.presets import GCS

        s = GCS(access_key="AK", secret_key="SK")
        assert s.endpoint == "https://storage.googleapis.com"

    def test_wasabi(self):
        from skyward.storage.presets import Wasabi

        s = Wasabi(region="eu-central-1", access_key="AK", secret_key="SK")
        assert s.endpoint == "https://s3.eu-central-1.wasabisys.com"

    def test_backblaze(self):
        from skyward.storage.presets import Backblaze

        s = Backblaze(region="us-west-004", key_id="KID", app_key="APPK")
        assert s.endpoint == "https://s3.us-west-004.backblazeb2.com"
        assert s.access_key == "KID"
        assert s.secret_key == "APPK"

    def test_hyperstack_returns_storage_with_callables(self):
        from skyward.storage import Storage
        from skyward.storage.presets import Hyperstack

        s = Hyperstack(api_key="test-key", region="CANADA-1")
        assert isinstance(s, Storage)
        assert s.endpoint == "https://ca1.obj.nexgencloud.io"
        assert callable(s.access_key)
        assert callable(s.secret_key)
        assert s.path_style is True

    def test_hyperstack_custom_endpoint(self):
        from skyward.storage.presets import Hyperstack

        s = Hyperstack(api_key="k", endpoint="https://custom.endpoint.io")
        assert s.endpoint == "https://custom.endpoint.io"

    def test_hyperstack_registers_on_close(self):
        from skyward.storage import _ON_CLOSE
        from skyward.storage.presets import Hyperstack

        s = Hyperstack(api_key="k")
        assert id(s) in _ON_CLOSE
        assert len(_ON_CLOSE[id(s)]) == 1
        _ON_CLOSE.pop(id(s))


class TestVolumeStorageField:
    def test_volume_accepts_storage(self):
        from skyward.api.spec import Volume
        from skyward.storage import Storage

        s = Storage(endpoint="https://abc.r2.cloudflarestorage.com", access_key="AK", secret_key="SK")
        v = Volume(bucket="b", mount="/data", storage=s)
        assert v.storage is s

    def test_volume_storage_default_none(self):
        from skyward.api.spec import Volume

        v = Volume(bucket="b", mount="/data")
        assert v.storage is None


class TestTopLevelExports:
    def test_storage_importable(self):
        import skyward as sky

        assert hasattr(sky, "Storage")

    def test_storage_in_all(self):
        import skyward

        assert "Storage" in skyward.__all__

    def test_storage_namespace(self):
        import skyward as sky

        assert hasattr(sky.storage, "R2")
        assert hasattr(sky.storage, "S3")
        assert hasattr(sky.storage, "GCS")
        assert hasattr(sky.storage, "Wasabi")
        assert hasattr(sky.storage, "Backblaze")
        assert hasattr(sky.storage, "Hyperstack")

    def test_volume_client_removed(self):
        import skyward

        assert "VolumeClient" not in skyward.__all__
