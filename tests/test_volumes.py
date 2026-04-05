from __future__ import annotations

import pytest

from skyward.core.spec import Volume
from skyward.providers.aws.config import AWS
from skyward.providers.provider import Mountable

pytestmark = [pytest.mark.xdist_group("unit")]


class TestVolume:
    def test_relative_mount_raises(self):
        with pytest.raises(ValueError, match="absolute path"):
            Volume(bucket="b", mount="data")

    def test_system_path_raises(self):
        for path in ("/", "/opt", "/opt/skyward", "/root", "/tmp"):
            with pytest.raises(ValueError, match="system path"):
                Volume(bucket="b", mount=path)



class TestMountVolumes:
    def test_single_volume_with_credentials(self):
        from skyward.providers.bootstrap.compose import resolve
        from skyward.providers.bootstrap.ops import mount_volumes
        from skyward.storage import Storage

        s = Storage(endpoint="https://storage.googleapis.com", access_key="AKIA123", secret_key="secret456")
        volumes = ((Volume(bucket="my-bucket", mount="/data", read_only=True), s),)
        script = resolve(mount_volumes(volumes))
        assert "s3fs" in script
        assert "AKIA123:secret456" in script
        assert "s3fs my-bucket /mnt/s3fs/my-bucket" in script
        assert "ln -sfn /mnt/s3fs/my-bucket /data" in script
        assert "url=https://storage.googleapis.com" in script
        assert "-o ro" in script

    def test_volume_read_write(self):
        from skyward.providers.bootstrap.compose import resolve
        from skyward.providers.bootstrap.ops import mount_volumes
        from skyward.storage import Storage

        s = Storage(endpoint="https://s3.amazonaws.com", access_key="a", secret_key="s")
        volumes = ((Volume(bucket="b", mount="/out", read_only=False), s),)
        script = resolve(mount_volumes(volumes))
        assert "-o ro" not in script

    def test_volume_with_prefix(self):
        from skyward.providers.bootstrap.compose import resolve
        from skyward.providers.bootstrap.ops import mount_volumes
        from skyward.storage import Storage

        s = Storage(endpoint="https://s3.amazonaws.com", access_key="a", secret_key="s")
        volumes = ((Volume(bucket="b", mount="/data", prefix="datasets/"), s),)
        script = resolve(mount_volumes(volumes))
        assert "s3fs b /mnt/s3fs/b" in script
        assert "ln -sfn /mnt/s3fs/b/datasets/ /data" in script

    def test_path_style_option(self):
        from skyward.providers.bootstrap.compose import resolve
        from skyward.providers.bootstrap.ops import mount_volumes
        from skyward.storage import Storage

        s = Storage(endpoint="https://ca1.obj.nexgencloud.io", access_key="ak", secret_key="sk", path_style=True)
        volumes = ((Volume(bucket="b", mount="/data"), s),)
        script = resolve(mount_volumes(volumes))
        assert "use_path_request_style" in script

    def test_no_path_style_by_default(self):
        from skyward.providers.bootstrap.compose import resolve
        from skyward.providers.bootstrap.ops import mount_volumes
        from skyward.storage import Storage

        s = Storage(endpoint="https://storage.googleapis.com", access_key="ak", secret_key="sk")
        volumes = ((Volume(bucket="b", mount="/data"), s),)
        script = resolve(mount_volumes(volumes))
        assert "use_path_request_style" not in script

    def test_iam_role_when_no_credentials(self):
        from skyward.providers.bootstrap.compose import resolve
        from skyward.providers.bootstrap.ops import mount_volumes
        from skyward.storage import Storage

        s = Storage(endpoint="https://s3.us-east-1.amazonaws.com")
        volumes = ((Volume(bucket="b", mount="/data"), s),)
        script = resolve(mount_volumes(volumes))
        assert "iam_role=auto" in script
        assert "/etc/s3fs-passwd" not in script

    def test_heterogeneous_storages(self):
        from skyward.providers.bootstrap.compose import resolve
        from skyward.providers.bootstrap.ops import mount_volumes
        from skyward.storage import Storage

        r2 = Storage(endpoint="https://abc.r2.cloudflarestorage.com", access_key="r2ak", secret_key="r2sk")
        s3 = Storage(endpoint="https://s3.us-east-1.amazonaws.com")
        volumes = (
            (Volume(bucket="data", mount="/data"), r2),
            (Volume(bucket="ckpt", mount="/ckpt"), s3),
        )
        script = resolve(mount_volumes(volumes))
        assert "r2ak:r2sk" in script
        assert "iam_role=auto" in script
        assert "url=https://abc.r2.cloudflarestorage.com" in script
        assert "url=https://s3.us-east-1.amazonaws.com" in script

    def test_same_bucket_same_storage_mounted_once(self):
        from skyward.providers.bootstrap.compose import resolve
        from skyward.providers.bootstrap.ops import mount_volumes
        from skyward.storage import Storage

        s = Storage(endpoint="https://s3.amazonaws.com")
        volumes = (
            (Volume(bucket="shared", mount="/data", prefix="datasets/", read_only=True), s),
            (Volume(bucket="shared", mount="/checkpoints", prefix="ckpt/", read_only=False), s),
        )
        script = resolve(mount_volumes(volumes))
        assert script.count("s3fs shared /mnt/s3fs/shared") == 1
        assert "-o ro" not in script
        assert "ln -sfn /mnt/s3fs/shared/datasets/ /data" in script
        assert "ln -sfn /mnt/s3fs/shared/ckpt/ /checkpoints" in script


class TestComputePoolVolumes:
    def test_pool_accepts_volumes(self):
        from skyward.core.pool import ComputePool

        vols = [Volume(bucket="b", mount="/data")]
        pool = ComputePool(provider=AWS(), volumes=vols)
        assert hasattr(pool, "volumes")
        assert len(pool.volumes) == 1
        assert pool.volumes[0].bucket == "b"


class TestBootstrapWithVolumes:
    def test_generate_bootstrap_with_volume_postamble(self):
        from skyward.core.spec import Image, generate_bootstrap
        from skyward.providers.bootstrap import mount_volumes, phase
        from skyward.storage import Storage

        image = Image(pip=["torch"])
        s = Storage(endpoint="https://s3.us-east-1.amazonaws.com")
        vols = ((Volume(bucket="my-data", mount="/data"), s),)

        postamble = phase("volumes", mount_volumes(vols))
        script = generate_bootstrap(image, ttl=0, postamble=postamble)

        assert "s3fs" in script
        assert "s3fs my-data /mnt/s3fs/my-data" in script
        assert "iam_role=auto" in script


class TestMountableValidation:
    def test_non_mountable_provider_does_not_implement_protocol(self):
        """Providers that don't implement Mountable should not satisfy the protocol."""
        from skyward.providers.container.config import Container

        assert not isinstance(Container(), Mountable)


