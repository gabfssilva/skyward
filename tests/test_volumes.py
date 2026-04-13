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
        assert "geesefs" in script
        assert "aws_access_key_id = AKIA123" in script
        assert "aws_secret_access_key = secret456" in script
        assert "my-bucket /mnt/geesefs/my-bucket" in script
        assert "ln -sfn /mnt/geesefs/my-bucket /data" in script
        assert "--endpoint=https://storage.googleapis.com" in script
        assert "-o allow_other,ro" in script

    def test_volume_read_write(self):
        from skyward.providers.bootstrap.compose import resolve
        from skyward.providers.bootstrap.ops import mount_volumes
        from skyward.storage import Storage

        s = Storage(endpoint="https://s3.amazonaws.com", access_key="a", secret_key="s")
        volumes = ((Volume(bucket="b", mount="/out", read_only=False), s),)
        script = resolve(mount_volumes(volumes))
        assert "allow_other,ro" not in script
        assert "-o allow_other" in script

    def test_volume_with_prefix(self):
        from skyward.providers.bootstrap.compose import resolve
        from skyward.providers.bootstrap.ops import mount_volumes
        from skyward.storage import Storage

        s = Storage(endpoint="https://s3.amazonaws.com", access_key="a", secret_key="s")
        volumes = ((Volume(bucket="b", mount="/data", prefix="datasets/"), s),)
        script = resolve(mount_volumes(volumes))
        assert "b /mnt/geesefs/b" in script
        assert "ln -sfn /mnt/geesefs/b/datasets/ /data" in script

    def test_path_style_option(self):
        from skyward.providers.bootstrap.compose import resolve
        from skyward.providers.bootstrap.ops import mount_volumes
        from skyward.storage import Storage

        s = Storage(endpoint="https://ca1.obj.nexgencloud.io", access_key="ak", secret_key="sk", path_style=True)
        volumes = ((Volume(bucket="b", mount="/data"), s),)
        script = resolve(mount_volumes(volumes))
        assert "--subdomain" not in script

    def test_no_path_style_by_default(self):
        from skyward.providers.bootstrap.compose import resolve
        from skyward.providers.bootstrap.ops import mount_volumes
        from skyward.storage import Storage

        s = Storage(endpoint="https://storage.googleapis.com", access_key="ak", secret_key="sk")
        volumes = ((Volume(bucket="b", mount="/data"), s),)
        script = resolve(mount_volumes(volumes))
        assert "--subdomain" in script

    def test_iam_role_when_no_credentials(self):
        from skyward.providers.bootstrap.compose import resolve
        from skyward.providers.bootstrap.ops import mount_volumes
        from skyward.storage import Storage

        s = Storage(endpoint="https://s3.us-east-1.amazonaws.com")
        volumes = ((Volume(bucket="b", mount="/data"), s),)
        script = resolve(mount_volumes(volumes))
        assert "--iam" in script
        assert "--iam-flavor=imdsv1" in script
        assert "/etc/geesefs-creds" not in script

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
        assert "aws_access_key_id = r2ak" in script
        assert "aws_secret_access_key = r2sk" in script
        assert "--iam" in script
        assert "--endpoint=https://abc.r2.cloudflarestorage.com" in script
        assert "--endpoint=https://s3.us-east-1.amazonaws.com" in script

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
        assert script.count("shared /mnt/geesefs/shared") == 1
        assert "allow_other,ro" not in script
        assert "ln -sfn /mnt/geesefs/shared/datasets/ /data" in script
        assert "ln -sfn /mnt/geesefs/shared/ckpt/ /checkpoints" in script


class TestComputePoolVolumes:
    def test_pool_accepts_volumes(self):
        from skyward.core.pool import ComputePool

        vols = [Volume(bucket="b", mount="/data")]
        pool = ComputePool(provider=AWS(), volumes=vols)
        assert hasattr(pool, "volumes")
        assert len(pool.volumes) == 1
        assert pool.volumes[0].bucket == "b"


class TestBootstrapWithVolumes:
    def test_generate_bootstrap_with_mount_plan_postamble(self):
        from skyward.api.model import MountPlan
        from skyward.core.spec import Image, generate_bootstrap
        from skyward.providers.bootstrap import mount_volumes, phase
        from skyward.storage import Storage

        image = Image(pip=["torch"])
        s = Storage(endpoint="https://s3.us-east-1.amazonaws.com")
        bootstrap_op = mount_volumes(((Volume(bucket="my-data", mount="/data"), s),))
        plan = MountPlan(bootstrap=bootstrap_op)
        assert plan.bootstrap is not None

        script = generate_bootstrap(image, ttl=0, postamble=phase("volumes", plan.bootstrap))

        assert "geesefs" in script
        assert "my-data /mnt/geesefs/my-data" in script
        assert "--iam" in script
        assert "volumes" in script  # phase marker present

    def test_generate_bootstrap_without_mount_plan_skips_volume_phase(self):
        from skyward.core.spec import Image, generate_bootstrap

        image = Image(pip=["torch"])
        script = generate_bootstrap(image, ttl=0, postamble=None)

        assert "geesefs" not in script
        assert "/mnt/geesefs" not in script
        # "volumes" appears in phase markers only when the op is wrapped; with postamble=None
        # no volumes-phase marker should be emitted.
        assert 'run_phase "volumes"' not in script
        assert 'emit_phase "started" "volumes"' not in script


class TestMountableValidation:
    def test_non_mountable_provider_does_not_implement_protocol(self):
        """Providers that don't implement Mountable should not satisfy the protocol."""
        from skyward.providers.container.config import Container

        assert not isinstance(Container(), Mountable)


class TestSymlinkVolumes:
    def test_multiple_prefixed_volumes_emit_mkdir_and_ln_only(self):
        from skyward.providers.bootstrap import symlink_volumes
        from skyward.providers.bootstrap.compose import resolve

        vols = (
            Volume(bucket="nv", mount="/data", prefix="datasets"),
            Volume(bucket="nv", mount="/ckpt", prefix="ckpt"),
        )
        script = resolve(symlink_volumes(vols, base="/workspace"))
        assert "mkdir -p /workspace/datasets" in script
        assert "ln -sfn /workspace/datasets /data" in script
        assert "mkdir -p /workspace/ckpt" in script
        assert "ln -sfn /workspace/ckpt /ckpt" in script
        assert "geesefs" not in script
        assert "apt-get" not in script

    def test_no_prefix_symlinks_base_directly(self):
        from skyward.providers.bootstrap import symlink_volumes
        from skyward.providers.bootstrap.compose import resolve

        script = resolve(symlink_volumes((Volume(bucket="b", mount="/data"),), base="/workspace"))
        assert "mkdir -p /workspace" in script
        assert "ln -sfn /workspace /data" in script


class TestFuseMountPlan:
    def test_renders_geesefs_script_with_endpoint_and_symlink(self):
        from skyward.providers.bootstrap import fuse_mount_plan
        from skyward.providers.bootstrap.compose import resolve
        from skyward.storage import Storage

        storage = Storage(endpoint="https://s3.us-east-1.amazonaws.com")
        plan = fuse_mount_plan((Volume(bucket="b", mount="/data"),), storage)

        assert plan.bootstrap is not None
        script = resolve(plan.bootstrap)
        assert "geesefs" in script
        assert "--endpoint=https://s3.us-east-1.amazonaws.com" in script
        assert "b /mnt/geesefs/b" in script
        assert "ln -sfn /mnt/geesefs/b /data" in script


class TestNativeMountPlan:
    def test_hints_propagate_and_bootstrap_is_symlink_only(self):
        from skyward.providers.bootstrap import native_mount_plan
        from skyward.providers.bootstrap.compose import resolve

        plan = native_mount_plan(
            (Volume(bucket="nv", mount="/data", prefix="sub"),),
            base="/workspace",
            networkVolumeId="vol-123",
            volumeMountPath="/workspace",
        )
        assert plan.deploy_hints["networkVolumeId"] == "vol-123"
        assert plan.deploy_hints["volumeMountPath"] == "/workspace"

        assert plan.bootstrap is not None
        script = resolve(plan.bootstrap)
        assert "ln -sfn /workspace/sub /data" in script
        assert "geesefs" not in script
        assert "apt-get" not in script

    def test_deploy_hints_are_immutable(self):
        import pytest

        from skyward.providers.bootstrap import native_mount_plan

        plan = native_mount_plan(
            (Volume(bucket="nv", mount="/data"),),
            base="/workspace",
            networkVolumeId="vol-1",
        )
        with pytest.raises(TypeError):
            plan.deploy_hints["networkVolumeId"] = "hacked"  # type: ignore[index]


def _minimal_runpod_cluster(*, mount_plan=None):
    """Build a Cluster[RunPodSpecific] for unit tests — no real API."""
    from skyward.api.spec import Nodes, PoolSpec
    from skyward.core.model import Cluster, InstanceType, Offer
    from skyward.providers.runpod.provider import RunPodSpecific

    spec = PoolSpec(nodes=Nodes(desired=1), accelerator=None, region="EU-RO-1")
    offer = Offer(
        id="o-1",
        instance_type=InstanceType(
            name="RTX_A6000", accelerator=None, vcpus=8, memory_gb=32,
            architecture="x86_64", specific=None,
        ),
        spot_price=0.5, on_demand_price=0.8, billing_unit="hour", specific=None,
    )
    specific = RunPodSpecific(gpu_type_id="NVIDIA RTX A6000", cloud_type="SECURE")
    return Cluster(
        id="c-1", status="provisioning", spec=spec, offer=offer,
        ssh_key_path="/tmp/key", ssh_user="root", use_sudo=False,
        shutdown_command="shutdown", specific=specific, mount_plan=mount_plan,
    )


class TestRunPodMountPlan:
    @staticmethod
    def _patch_list(monkeypatch, volumes):
        from skyward.providers.runpod.client import RunPodClient

        async def fake_list(self):  # noqa: ANN001
            return volumes

        monkeypatch.setattr(RunPodClient, "list_network_volumes", fake_list)

    @pytest.mark.asyncio
    async def test_volume_bucket_resolved_by_id(self, monkeypatch):
        from skyward.providers.bootstrap.compose import resolve
        from skyward.providers.runpod.config import RunPod
        from skyward.providers.runpod.provider import RunPodProvider

        self._patch_list(monkeypatch, [
            {"id": "aqsojarpxt", "name": "my-volume", "dataCenterId": "EU-RO-1"},
        ])
        config = RunPod(api_key="k", data_center_ids=("EU-RO-1",))
        provider = await RunPodProvider.create(config)
        cluster = _minimal_runpod_cluster()

        plan = await provider.mount_plan(
            cluster, (Volume(bucket="aqsojarpxt", mount="/data", prefix="sub"),),
        )
        assert plan.deploy_hints["networkVolumeId"] == "aqsojarpxt"
        assert plan.bootstrap is not None
        assert "ln -sfn /workspace/sub /data" in resolve(plan.bootstrap)

    @pytest.mark.asyncio
    async def test_volume_bucket_resolved_by_name(self, monkeypatch):
        from skyward.providers.runpod.config import RunPod
        from skyward.providers.runpod.provider import RunPodProvider

        self._patch_list(monkeypatch, [
            {"id": "aqsojarpxt", "name": "my-volume", "dataCenterId": "EU-RO-1"},
            {"id": "other-id", "name": "other", "dataCenterId": "EU-RO-1"},
        ])
        config = RunPod(api_key="k", data_center_ids=("EU-RO-1",))
        provider = await RunPodProvider.create(config)
        cluster = _minimal_runpod_cluster()

        plan = await provider.mount_plan(cluster, (Volume(bucket="my-volume", mount="/data"),))
        # Resolved to the matching id, not the name
        assert plan.deploy_hints["networkVolumeId"] == "aqsojarpxt"

    @pytest.mark.asyncio
    async def test_unknown_name_raises_with_available_list(self, monkeypatch):
        from skyward.providers.runpod.config import RunPod
        from skyward.providers.runpod.provider import RunPodProvider

        self._patch_list(monkeypatch, [
            {"id": "aqsojarpxt", "name": "my-volume", "dataCenterId": "EU-RO-1"},
        ])
        config = RunPod(api_key="k", data_center_ids=("EU-RO-1",))
        provider = await RunPodProvider.create(config)
        cluster = _minimal_runpod_cluster()

        with pytest.raises(ValueError) as excinfo:
            await provider.mount_plan(cluster, (Volume(bucket="nonexistent", mount="/data"),))
        assert "not found" in str(excinfo.value)
        # actionable: lists the available volumes
        assert "my-volume" in str(excinfo.value)
        assert "aqsojarpxt" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_cross_dc_volume_rejected(self, monkeypatch):
        from skyward.providers.runpod.config import RunPod
        from skyward.providers.runpod.provider import RunPodProvider

        self._patch_list(monkeypatch, [
            {"id": "nv-us", "name": "us-vol", "dataCenterId": "US-CA-2"},
        ])
        config = RunPod(api_key="k", data_center_ids=("EU-RO-1",))
        provider = await RunPodProvider.create(config)
        cluster = _minimal_runpod_cluster()

        with pytest.raises(ValueError, match="cross-DC"):
            await provider.mount_plan(cluster, (Volume(bucket="us-vol", mount="/data"),))

    @pytest.mark.asyncio
    async def test_multiple_volumes_same_bucket_project_via_prefix(self, monkeypatch):
        from skyward.providers.bootstrap.compose import resolve
        from skyward.providers.runpod.config import RunPod
        from skyward.providers.runpod.provider import RunPodProvider

        self._patch_list(monkeypatch, [
            {"id": "aqsojarpxt", "name": "my-volume", "dataCenterId": "EU-RO-1"},
        ])
        config = RunPod(api_key="k", data_center_ids=("EU-RO-1",))
        provider = await RunPodProvider.create(config)
        cluster = _minimal_runpod_cluster()

        plan = await provider.mount_plan(
            cluster,
            (
                Volume(bucket="my-volume", mount="/data", prefix="datasets"),
                Volume(bucket="my-volume", mount="/ckpt", prefix="ckpt"),
            ),
        )
        assert plan.deploy_hints["networkVolumeId"] == "aqsojarpxt"
        assert plan.bootstrap is not None
        script = resolve(plan.bootstrap)
        assert "ln -sfn /workspace/datasets /data" in script
        assert "ln -sfn /workspace/ckpt /ckpt" in script

    @pytest.mark.asyncio
    async def test_multiple_volumes_different_buckets_rejected(self):
        from skyward.providers.runpod.config import RunPod
        from skyward.providers.runpod.provider import RunPodProvider

        config = RunPod(api_key="k", data_center_ids=("EU-RO-1",))
        provider = await RunPodProvider.create(config)
        cluster = _minimal_runpod_cluster()

        with pytest.raises(ValueError, match="one network volume per pod"):
            await provider.mount_plan(
                cluster,
                (
                    Volume(bucket="nv-a", mount="/a"),
                    Volume(bucket="nv-b", mount="/b"),
                ),
            )

    @pytest.mark.asyncio
    async def test_empty_bucket_raises_with_runpodctl_hint(self):
        from skyward.providers.runpod.config import RunPod
        from skyward.providers.runpod.provider import RunPodProvider

        config = RunPod(api_key="k", data_center_ids=("EU-RO-1",))
        provider = await RunPodProvider.create(config)
        cluster = _minimal_runpod_cluster()

        with pytest.raises(ValueError, match="id or name") as excinfo:
            await provider.mount_plan(cluster, (Volume(bucket="", mount="/data"),))
        assert "runpodctl" in str(excinfo.value)


class TestCreateGpuPodRestPayload:
    """_create_gpu_pod_rest should propagate deploy_hints into PodCreateParams."""

    @pytest.mark.asyncio
    async def test_rest_payload_carries_network_volume_id(self, monkeypatch):
        from skyward.api.model import MountPlan
        from skyward.providers.runpod.client import RunPodClient
        from skyward.providers.runpod.config import RunPod
        from skyward.providers.runpod.provider import _create_gpu_pod_rest

        captured: dict[str, object] = {}

        async def fake_create_pod(self, params):  # noqa: ANN001
            captured["params"] = params
            return {"id": "pod-1", "imageName": "img", "desiredStatus": "RUNNING"}

        monkeypatch.setattr(RunPodClient, "create_pod", fake_create_pod)

        config = RunPod(api_key="k", data_center_ids=("EU-RO-1",))
        plan = MountPlan(deploy_hints={"networkVolumeId": "nv-1", "volumeMountPath": "/mnt/nv"})
        cluster = _minimal_runpod_cluster(mount_plan=plan)

        async with RunPodClient(api_key="k", config=config) as client:
            await _create_gpu_pod_rest(
                client, config, cluster, node_index=0, ssh_public_key="k",
                image_name="img", use_spot=False,
            )

        params = captured["params"]
        assert params["networkVolumeId"] == "nv-1"
        assert params["volumeMountPath"] == "/mnt/nv"

    @pytest.mark.asyncio
    async def test_rest_payload_omits_network_volume_id_when_no_plan(self, monkeypatch):
        from skyward.providers.runpod.client import RunPodClient
        from skyward.providers.runpod.config import RunPod
        from skyward.providers.runpod.provider import _create_gpu_pod_rest

        captured: dict[str, object] = {}

        async def fake_create_pod(self, params):  # noqa: ANN001
            captured["params"] = params
            return {"id": "pod-1", "imageName": "img", "desiredStatus": "RUNNING"}

        monkeypatch.setattr(RunPodClient, "create_pod", fake_create_pod)

        config = RunPod(api_key="k", data_center_ids=("EU-RO-1",))
        cluster = _minimal_runpod_cluster(mount_plan=None)

        async with RunPodClient(api_key="k", config=config) as client:
            await _create_gpu_pod_rest(
                client, config, cluster, node_index=0, ssh_public_key="k",
                image_name="img", use_spot=False,
            )

        assert "networkVolumeId" not in captured["params"]  # type: ignore[operator]


class TestBuildMountPlan:
    @pytest.mark.asyncio
    async def test_all_explicit_storage_builds_fuse_plan_even_for_non_mountable_provider(self):
        from skyward.actors.pool.actor import _build_mount_plan
        from skyward.providers.bootstrap.compose import resolve
        from skyward.storage import Storage

        class _StubStorage:
            async def resolve(self):  # noqa: ANN202
                return Storage(endpoint="https://s3.us-east-1.amazonaws.com")

        vols = (Volume(bucket="b", mount="/data", storage=_StubStorage()),)  # type: ignore[arg-type]
        plan = await _build_mount_plan(vols, cluster=None, provider=object())

        assert plan.deploy_hints == {}
        script = resolve(plan.bootstrap)
        assert "geesefs" in script
        assert "b /mnt/geesefs/b" in script

    @pytest.mark.asyncio
    async def test_no_storage_non_mountable_raises_with_bucket_names(self):
        from skyward.actors.pool.actor import _build_mount_plan

        vols = (Volume(bucket="my-bucket", mount="/data"),)

        with pytest.raises(RuntimeError, match="my-bucket"):
            await _build_mount_plan(vols, cluster=None, provider=object())

    @pytest.mark.asyncio
    async def test_delegates_to_mountable_provider(self):
        from skyward.actors.pool.actor import _build_mount_plan
        from skyward.api.model import MountPlan

        called_with: dict[str, object] = {}
        returned = MountPlan(deploy_hints={"x": "y"})

        class _StubMountableProvider:
            async def mount_plan(self, cluster, volumes):  # noqa: ANN001
                called_with["cluster"] = cluster
                called_with["volumes"] = volumes
                return returned

        vols = (Volume(bucket="b", mount="/data"),)
        plan = await _build_mount_plan(vols, cluster="CLUSTER", provider=_StubMountableProvider())

        assert plan is returned
        assert called_with["cluster"] == "CLUSTER"
        assert called_with["volumes"] == vols

    @pytest.mark.asyncio
    async def test_mixed_explicit_and_provider_managed_rejected(self):
        from skyward.actors.pool.actor import _build_mount_plan
        from skyward.storage import Storage

        class _StubStorage:
            async def resolve(self):  # noqa: ANN202
                return Storage(endpoint="https://s3.us-east-1.amazonaws.com")

        class _StubMountableProvider:
            async def mount_plan(self, cluster, volumes):  # noqa: ANN001, ANN202
                raise AssertionError("should not be called")

        vols = (
            Volume(bucket="a", mount="/a", storage=_StubStorage()),  # type: ignore[arg-type]
            Volume(bucket="b", mount="/b"),
        )
        with pytest.raises(RuntimeError, match="Mixing"):
            await _build_mount_plan(vols, cluster=None, provider=_StubMountableProvider())


class TestDeployGpuPodPayload:
    """GraphQL payload should carry networkVolumeId when requested."""

    @pytest.mark.asyncio
    async def test_graphql_input_includes_network_volume_id(self, monkeypatch):
        from skyward.providers.runpod.client import RunPodClient
        from skyward.providers.runpod.config import RunPod

        captured: dict[str, object] = {}

        async def fake_graphql(self, query, variables=None):  # noqa: ANN001
            captured["variables"] = variables
            return {"podFindAndDeployOnDemand": {"id": "pod-1"}}

        monkeypatch.setattr(RunPodClient, "_graphql", fake_graphql)

        async with RunPodClient(api_key="k", config=RunPod()) as client:
            await client.deploy_gpu_pod(
                name="n", image_name="img", gpu_type_id="t",
                network_volume_id="nv-xyz",
                volume_mount_path="/workspace",
            )

        variables = captured["variables"]
        assert isinstance(variables, dict)
        input_vars = variables["input"]
        assert input_vars["networkVolumeId"] == "nv-xyz"
        assert input_vars["volumeMountPath"] == "/workspace"

    @pytest.mark.asyncio
    async def test_graphql_omits_network_volume_id_when_none(self, monkeypatch):
        from skyward.providers.runpod.client import RunPodClient
        from skyward.providers.runpod.config import RunPod

        captured: dict[str, object] = {}

        async def fake_graphql(self, query, variables=None):  # noqa: ANN001
            captured["variables"] = variables
            return {"podFindAndDeployOnDemand": {"id": "pod-1"}}

        monkeypatch.setattr(RunPodClient, "_graphql", fake_graphql)

        async with RunPodClient(api_key="k", config=RunPod()) as client:
            await client.deploy_gpu_pod(name="n", image_name="img", gpu_type_id="t")

        input_vars = captured["variables"]["input"]  # type: ignore[index]
        assert "networkVolumeId" not in input_vars


