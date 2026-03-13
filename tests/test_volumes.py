from __future__ import annotations

import pytest

import skyward as sky
from skyward.api.pool import Pool
from skyward.core.spec import Nodes, PoolSpec, Volume
from skyward.providers.aws.config import AWS
from skyward.providers.provider import Mountable

pytestmark = [pytest.mark.xdist_group("unit")]


class TestVolume:
    def test_basic_creation(self):
        v = Volume(bucket="my-bucket", mount="/data")
        assert v.bucket == "my-bucket"
        assert v.mount == "/data"
        assert v.prefix == ""
        assert v.read_only is True

    def test_with_prefix_and_read_write(self):
        v = Volume(bucket="b", mount="/checkpoints", prefix="exp/", read_only=False)
        assert v.prefix == "exp/"
        assert v.read_only is False

    def test_relative_mount_raises(self):
        with pytest.raises(ValueError, match="absolute path"):
            Volume(bucket="b", mount="data")

    def test_system_path_raises(self):
        for path in ("/", "/opt", "/opt/skyward", "/root", "/tmp"):
            with pytest.raises(ValueError, match="system path"):
                Volume(bucket="b", mount=path)

    def test_frozen(self):
        v = Volume(bucket="b", mount="/data")
        with pytest.raises(AttributeError):
            v.bucket = "other"  # type: ignore[misc]

    def test_poolspec_volumes_default_empty(self):
        spec = PoolSpec(nodes=Nodes(min=1), accelerator=None, region="us-east-1")
        assert spec.volumes == ()

    def test_poolspec_with_volumes(self):
        vols = (Volume(bucket="b", mount="/data"),)
        spec = PoolSpec(nodes=Nodes(min=1), accelerator=None, region="us-east-1", volumes=vols)
        assert len(spec.volumes) == 1
        assert spec.volumes[0].mount == "/data"


class TestStorageReplacedMountEndpoint:
    def test_storage_with_credentials(self):
        from skyward.storage import Storage

        s = Storage(
            endpoint="https://storage.googleapis.com",
            access_key="AKIA...",
            secret_key="secret",
        )
        assert s.endpoint == "https://storage.googleapis.com"
        assert s.access_key == "AKIA..."

    def test_storage_without_credentials(self):
        from skyward.storage import Storage

        s = Storage(endpoint="https://s3.us-east-1.amazonaws.com")
        assert s.access_key is None

    def test_mountable_is_runtime_checkable(self):
        from skyward.providers.provider import Mountable

        assert isinstance(Mountable, type)

    def test_mountable_has_storage_method(self):
        from skyward.providers.provider import Mountable

        assert hasattr(Mountable, "storage")


class TestClusterResolvedVolumes:
    def test_resolved_volumes_field_default_none(self):
        import dataclasses
        from skyward.core.model import Cluster

        fields = {f.name: f for f in dataclasses.fields(Cluster)}
        assert "resolved_volumes" in fields
        assert fields["resolved_volumes"].default is None

    def test_mount_endpoint_removed(self):
        import dataclasses
        from skyward.core.model import Cluster

        fields = {f.name for f in dataclasses.fields(Cluster)}
        assert "mount_endpoint" not in fields


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

    def test_pool_default_no_volumes(self):
        from skyward.core.pool import ComputePool

        pool = ComputePool(provider=AWS())
        assert pool.volumes == ()

    def test_pool_converts_list_to_tuple(self):
        from skyward.core.pool import ComputePool

        vols = [Volume(bucket="b", mount="/data")]
        pool = ComputePool(provider=AWS(), volumes=vols)
        assert isinstance(pool.volumes, tuple)


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


class TestAWSMountable:
    def test_aws_provider_has_storage(self):
        from skyward.providers.aws.provider import AWSProvider
        assert hasattr(AWSProvider, "storage")


class TestGCPMountable:
    def test_gcp_provider_has_storage(self):
        from skyward.providers.gcp.provider import GCPProvider
        assert hasattr(GCPProvider, "storage")


class TestRunPodMountable:
    def test_runpod_provider_has_storage(self):
        from skyward.providers.runpod.provider import RunPodProvider
        assert hasattr(RunPodProvider, "storage")

    def test_runpod_s3_datacenters_defined(self):
        from skyward.providers.runpod.provider import _RUNPOD_S3_DATACENTERS
        assert len(_RUNPOD_S3_DATACENTERS) == 11


class TestExports:
    def test_volume_importable_from_skyward(self):
        import skyward as sky
        assert hasattr(sky, "Volume")

    def test_volume_in_all(self):
        import skyward
        assert "Volume" in skyward.__all__
