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

    def test_frozen(self):
        from skyward.storage import Storage

        s = Storage(endpoint="https://s3.amazonaws.com")
        with pytest.raises(AttributeError):
            s.endpoint = "other"  # type: ignore[misc]


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

    def test_volume_client_removed(self):
        import skyward

        assert "VolumeClient" not in skyward.__all__
