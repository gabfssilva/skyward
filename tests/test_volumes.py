from __future__ import annotations

import pytest

from skyward.api.pool import ComputePool
from skyward.api.spec import PoolSpec, Volume
from skyward.providers.aws.config import AWS
from skyward.providers.provider import Mountable, MountEndpoint

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
        spec = PoolSpec(nodes=1, accelerator=None, region="us-east-1")
        assert spec.volumes == ()

    def test_poolspec_with_volumes(self):
        vols = (Volume(bucket="b", mount="/data"),)
        spec = PoolSpec(nodes=1, accelerator=None, region="us-east-1", volumes=vols)
        assert len(spec.volumes) == 1
        assert spec.volumes[0].mount == "/data"


class TestMountEndpoint:
    def test_with_credentials(self):
        ep = MountEndpoint(
            endpoint="https://storage.googleapis.com",
            access_key="AKIA...",
            secret_key="secret",
        )
        assert ep.endpoint == "https://storage.googleapis.com"
        assert ep.access_key == "AKIA..."
        assert ep.secret_key == "secret"

    def test_without_credentials(self):
        ep = MountEndpoint(endpoint="https://s3.us-east-1.amazonaws.com")
        assert ep.access_key is None
        assert ep.secret_key is None

    def test_frozen(self):
        ep = MountEndpoint(endpoint="https://s3.amazonaws.com")
        with pytest.raises(AttributeError):
            ep.endpoint = "other"  # type: ignore[misc]

    def test_mountable_is_runtime_checkable(self):
        assert isinstance(Mountable, type)


class TestClusterMountEndpoint:
    def test_mount_endpoint_field_exists_with_default_none(self):
        import dataclasses

        from skyward.api.model import Cluster
        fields = {f.name: f for f in dataclasses.fields(Cluster)}
        assert "mount_endpoint" in fields
        assert fields["mount_endpoint"].default is None


class TestMountVolumes:
    def test_single_volume_with_credentials(self):
        from skyward.providers.bootstrap.compose import resolve
        from skyward.providers.bootstrap.ops import mount_volumes

        volumes = (Volume(bucket="my-bucket", mount="/data", read_only=True),)
        endpoint = MountEndpoint(
            endpoint="https://storage.googleapis.com",
            access_key="AKIA123",
            secret_key="secret456",
        )
        script = resolve(mount_volumes(volumes, endpoint))
        assert "s3fs" in script
        assert "AKIA123:secret456" in script
        assert "/etc/s3fs-passwd" in script
        assert "chmod 600" in script
        assert "mkdir -p /data" in script
        assert "s3fs my-bucket" in script
        assert "url=https://storage.googleapis.com" in script
        assert "-o ro" in script

    def test_volume_read_write(self):
        from skyward.providers.bootstrap.compose import resolve
        from skyward.providers.bootstrap.ops import mount_volumes

        volumes = (Volume(bucket="b", mount="/out", read_only=False),)
        endpoint = MountEndpoint(
            endpoint="https://s3.amazonaws.com", access_key="a", secret_key="s",
        )
        script = resolve(mount_volumes(volumes, endpoint))
        assert "-o ro" not in script

    def test_volume_with_prefix(self):
        from skyward.providers.bootstrap.compose import resolve
        from skyward.providers.bootstrap.ops import mount_volumes

        volumes = (Volume(bucket="b", mount="/data", prefix="datasets/"),)
        endpoint = MountEndpoint(
            endpoint="https://s3.amazonaws.com", access_key="a", secret_key="s",
        )
        script = resolve(mount_volumes(volumes, endpoint))
        assert "b:/datasets/" in script

    def test_iam_role_when_no_credentials(self):
        from skyward.providers.bootstrap.compose import resolve
        from skyward.providers.bootstrap.ops import mount_volumes

        volumes = (Volume(bucket="b", mount="/data"),)
        endpoint = MountEndpoint(endpoint="https://s3.us-east-1.amazonaws.com")
        script = resolve(mount_volumes(volumes, endpoint))
        assert "iam_role=auto" in script
        assert "/etc/s3fs-passwd" not in script

    def test_multiple_volumes(self):
        from skyward.providers.bootstrap.compose import resolve
        from skyward.providers.bootstrap.ops import mount_volumes

        volumes = (
            Volume(bucket="data-bucket", mount="/data", read_only=True),
            Volume(bucket="ckpt-bucket", mount="/checkpoints", read_only=False),
        )
        endpoint = MountEndpoint(
            endpoint="https://s3.amazonaws.com",
            access_key="ak",
            secret_key="sk",
        )
        script = resolve(mount_volumes(volumes, endpoint))
        assert "mkdir -p /data" in script
        assert "mkdir -p /checkpoints" in script
        assert "s3fs data-bucket" in script
        assert "s3fs ckpt-bucket" in script


class TestComputePoolVolumes:
    def test_pool_accepts_volumes(self):
        vols = [Volume(bucket="b", mount="/data")]
        pool = ComputePool(provider=AWS(), volumes=vols)
        assert hasattr(pool, "volumes")
        assert len(pool.volumes) == 1
        assert pool.volumes[0].bucket == "b"

    def test_pool_default_no_volumes(self):
        pool = ComputePool(provider=AWS())
        assert pool.volumes == ()

    def test_pool_converts_list_to_tuple(self):
        vols = [Volume(bucket="b", mount="/data")]
        pool = ComputePool(provider=AWS(), volumes=vols)
        assert isinstance(pool.volumes, tuple)


class TestBootstrapWithVolumes:
    def test_generate_bootstrap_with_volume_postamble(self):
        """Verify Image.generate_bootstrap includes volume mount ops via postamble."""
        from skyward.api.spec import Image
        from skyward.providers.bootstrap import mount_volumes, phase

        image = Image(pip=["torch"])
        vols = (Volume(bucket="my-data", mount="/data"),)
        endpoint = MountEndpoint(endpoint="https://s3.us-east-1.amazonaws.com")

        postamble = phase("volumes", mount_volumes(vols, endpoint))
        script = image.generate_bootstrap(ttl=0, postamble=postamble)

        assert "s3fs" in script
        assert "mkdir -p /data" in script
        assert "s3fs my-data" in script
        assert "iam_role=auto" in script


class TestMountableValidation:
    def test_non_mountable_provider_does_not_implement_protocol(self):
        """Providers that don't implement Mountable should not satisfy the protocol."""
        from skyward.providers.container.config import Container

        assert not isinstance(Container(), Mountable)


class TestAWSMountable:
    def test_aws_provider_has_mount_endpoint(self):
        from skyward.providers.aws.provider import AWSProvider
        assert hasattr(AWSProvider, "mount_endpoint")

    def test_aws_provider_is_mountable(self):
        from skyward.providers.aws.provider import AWSProvider
        assert hasattr(AWSProvider, "mount_endpoint")


class TestGCPMountable:
    def test_gcp_provider_has_mount_endpoint(self):
        from skyward.providers.gcp.provider import GCPProvider
        assert hasattr(GCPProvider, "mount_endpoint")


class TestRunPodMountable:
    def test_runpod_provider_has_mount_endpoint(self):
        from skyward.providers.runpod.provider import RunPodProvider
        assert hasattr(RunPodProvider, "mount_endpoint")

    def test_runpod_s3_datacenters_defined(self):
        from skyward.providers.runpod.provider import _RUNPOD_S3_DATACENTERS
        assert len(_RUNPOD_S3_DATACENTERS) == 11
        assert "US-CA-2" in _RUNPOD_S3_DATACENTERS
        assert "EUR-IS-1" in _RUNPOD_S3_DATACENTERS


class TestExports:
    def test_volume_importable_from_skyward(self):
        import skyward as sky
        assert hasattr(sky, "Volume")

    def test_volume_in_all(self):
        import skyward
        assert "Volume" in skyward.__all__
