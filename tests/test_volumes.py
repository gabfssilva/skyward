"""Mounting a bucket: what the client refuses, what the shell says, and who signs for it.

The shell is the whole feature. A mount plan that reads correctly and renders a
``geesefs`` line with the wrong flag is a compute that boots, reports ready, and
hands the training loop an empty directory — so every credential path is asserted
on the text it actually produces.
"""

from __future__ import annotations

import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from skyward.server.application import market
from skyward.shared.errors import CapabilityMismatchError
from skyward.server.application.machines import Machines
from skyward.shared.provider import Binding, Machine, Mount, Mountable
from skyward.server.persistence.computes import ComputeStore
from skyward.server.persistence.db import connect
from skyward.server.persistence.functions import BlobStore
from skyward.shared.schemas import ComputeCreate, ComputeSpec, Endpoint, NodeBounds, Offer, ProviderRef, Spec, Volume
from skyward.providers.aws import AWSProvider
from skyward.providers.runpod import RunPodProvider
from skyward.worker import bootstrap
from skyward.core.spec import Volume as ClientVolume

pytestmark = pytest.mark.unit

OFFER = Offer(
    id="ofr_1",
    provider_id="prv_1",
    provider_name="fake",
    kind="fake",
    instance_type="t3.micro",
    accelerator_count=0,
    cpus=2,
    memory_gb=4.0,
    region="us-east-1",
    on_demand_price=0.05,
    fetched_at=datetime.now(UTC),
    expires_at=datetime.now(UTC) + timedelta(hours=1),
)


def spec(*volumes: Volume) -> ComputeSpec:
    return ComputeSpec(
        specs=(Spec(provider=ProviderRef(kind="fake"), cpus=1, memory_gb=1),),
        nodes=NodeBounds(desired=1),
        volumes=volumes,
    )


class Deaf:
    """A provider that cannot mount anything, which is most of them."""

    kind = "deaf"

    def allows_cluster_formation(self, spec: ComputeSpec, offer: Offer) -> bool:
        return True

    async def initialize(self, compute_id: str, spec: ComputeSpec, offer: Offer, market: str, public_key: str) -> Binding:
        return {"region": "us-east-1"}


class Attaching:
    """A provider that satisfies volumes by attaching storage before the machine boots."""

    kind = "attaching"

    def allows_cluster_formation(self, spec: ComputeSpec, offer: Offer) -> bool:
        return True

    async def initialize(self, compute_id: str, spec: ComputeSpec, offer: Offer, market: str, public_key: str) -> Binding:
        return {"region": "us-east-1"}

    async def mount(self, binding: Binding, volumes: tuple[Volume, ...]) -> Mount:
        return Mount(binding_patch={"attachment": "vol-7"}, phases=(bootstrap.symlinks(volumes, "/workspace"),))

    async def launch(self, binding: Binding, market: str, count: int, min_count: int) -> tuple[Binding, list[Machine]]:
        return binding, [Machine(id="m-0", state="running", host="10.0.0.1")]


class Scripted(Machines):
    def __init__(self, computes: ComputeStore, blobs: BlobStore, adapter: object) -> None:
        super().__init__(computes, None, None, None, blobs)  # type: ignore[arg-type]
        self._adapter = adapter

    async def adapter(self, provider_id: str | None) -> Any:  # type: ignore[override]
        return self._adapter


@pytest.fixture
async def store(tmp_path: Path) -> tuple[ComputeStore, BlobStore]:
    await connect(tmp_path / "skyward.sqlite")
    return ComputeStore(), BlobStore()


@pytest.fixture(autouse=True)
def _one_offer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bind picks the only machine on sale, so the test is about the mount and not the market."""

    async def pick(_: ComputeSpec, __: object) -> tuple[Offer, str]:
        return OFFER, "on_demand"

    monkeypatch.setattr(market, "pick", pick)


# --------------------------------------------------------------------------- #
# What the client refuses before anything is provisioned
# --------------------------------------------------------------------------- #


def test_a_volume_must_mount_at_an_absolute_path():
    with pytest.raises(ValueError, match="absolute path"):
        ClientVolume(bucket="data", mount="data")


@pytest.mark.parametrize("mount", ["/", "/opt", "/opt/skyward", "/root", "/tmp"])
def test_a_volume_cannot_mount_over_the_machine_itself(mount: str):
    """``/opt/skyward`` is the venv and the journal: a bucket there is a node that stops reporting."""
    with pytest.raises(ValueError, match="machine's own"):
        ClientVolume(bucket="data", mount=mount)


def test_a_trailing_slash_does_not_smuggle_a_system_path_past_the_check():
    with pytest.raises(ValueError, match="machine's own"):
        ClientVolume(bucket="data", mount="/opt/skyward/")


def test_an_ordinary_path_is_accepted():
    volume = ClientVolume(bucket="data", mount="/data")
    assert volume.read_only, "a network filesystem is read-heavy, so read-only is the default"


# --------------------------------------------------------------------------- #
# The shell, per credential path
# --------------------------------------------------------------------------- #


def test_a_bucket_with_no_access_key_is_signed_for_by_the_machine_itself():
    """The AWS path: the instance profile signs, so no secret is minted and none travels."""
    text = bootstrap.mounts(((Volume(bucket="training", mount="/data"), Endpoint(url="https://s3.us-east-1.amazonaws.com")),))
    assert "--iam --iam-flavor=imdsv1" in text
    assert "--shared-config" not in text
    assert "aws_access_key_id" not in text


def test_an_endpoint_with_keys_writes_one_locked_down_credentials_file():
    text = bootstrap.mounts((
        (Volume(bucket="training", mount="/data"), Endpoint(url="https://storage.googleapis.com", access_key="AK", secret_key="SK")),
    ))
    assert "[default]" in text
    assert "aws_access_key_id = AK" in text
    assert "aws_secret_access_key = SK" in text
    assert "chmod 600 /etc/geesefs-creds-" in text
    assert "--shared-config=/etc/geesefs-creds-" in text
    assert "--iam" not in text


def test_two_buckets_on_one_endpoint_share_a_single_credentials_file():
    """The file is keyed by endpoint, so ten buckets in one account cost one write."""
    endpoint = Endpoint(url="https://storage.googleapis.com", access_key="AK", secret_key="SK")
    text = bootstrap.mounts((
        (Volume(bucket="a", mount="/a"), endpoint),
        (Volume(bucket="b", mount="/b"), endpoint),
    ))
    assert text.count("aws_access_key_id") == 1


def test_a_path_style_endpoint_is_not_asked_to_serve_buckets_as_subdomains():
    """The Hyperstack path: its endpoint has no per-bucket hostname, so ``--subdomain`` would 404."""
    text = bootstrap.mounts((
        (Volume(bucket="training", mount="/data"), Endpoint(url="https://ca1.obj.nexgencloud.io", access_key="AK", secret_key="SK", path_style=True)),
    ))
    assert "--subdomain" not in text


def test_a_virtual_hosted_endpoint_asks_for_subdomain_addressing():
    text = bootstrap.mounts(((Volume(bucket="training", mount="/data"), Endpoint(url="https://s3.us-east-1.amazonaws.com")),))
    assert "--subdomain" in text


def test_a_read_only_volume_is_mounted_ro_and_a_writable_one_is_not():
    read = bootstrap.mounts(((Volume(bucket="a", mount="/a"), Endpoint(url="https://s3.amazonaws.com")),))
    write = bootstrap.mounts(((Volume(bucket="a", mount="/a", read_only=False), Endpoint(url="https://s3.amazonaws.com")),))
    assert "-o allow_other,ro" in read
    assert "-o allow_other " in write and ",ro" not in write


def test_a_bucket_wanted_writable_anywhere_is_mounted_writable_everywhere():
    """One bucket is one mount, so the narrower of two intents would fail the write that asked."""
    endpoint = Endpoint(url="https://s3.amazonaws.com")
    text = bootstrap.mounts((
        (Volume(bucket="shared", mount="/read"), endpoint),
        (Volume(bucket="shared", mount="/write", read_only=False), endpoint),
    ))
    assert ",ro" not in text


def test_one_bucket_is_mounted_once_however_many_volumes_name_it():
    endpoint = Endpoint(url="https://s3.amazonaws.com")
    text = bootstrap.mounts((
        (Volume(bucket="shared", mount="/one", prefix="a"), endpoint),
        (Volume(bucket="shared", mount="/two", prefix="b"), endpoint),
    ))
    assert text.count(f"{bootstrap.GEESEFS_BIN} --endpoint") == 1
    assert f"ln -sfn {bootstrap.FUSE_ROOT}/shared/a /one" in text
    assert f"ln -sfn {bootstrap.FUSE_ROOT}/shared/b /two" in text


def test_a_prefix_on_a_read_only_bucket_is_not_created_before_it_is_linked():
    """``mkdir`` inside a read-only mount fails, and a failing command takes the phase with it."""
    text = bootstrap.mounts(((Volume(bucket="a", mount="/data", prefix="train"), Endpoint(url="https://s3.amazonaws.com")),))
    assert f"mkdir -p {bootstrap.FUSE_ROOT}/a/train" not in text
    assert f"ln -sfn {bootstrap.FUSE_ROOT}/a/train /data" in text


def test_the_fuse_fallback_is_grouped_so_it_cannot_swallow_the_install_before_it():
    """``a && b || c && d`` binds left: ungrouped, a failed ``ca-certificates`` reaches ``d`` anyway."""
    text = bootstrap.mounts(((Volume(bucket="a", mount="/a"), Endpoint(url="https://s3.amazonaws.com")),))
    assert "{ apt-get" in text and "install -y -qq fuse; }" in text


def test_the_geesefs_binary_is_pinned_rather_than_pulled_from_latest():
    """An unpinned download is a different filesystem driver on every boot."""
    text = bootstrap.mounts(((Volume(bucket="a", mount="/a"), Endpoint(url="https://s3.amazonaws.com")),))
    assert f"/releases/download/{bootstrap.GEESEFS}/geesefs-linux-" in text
    assert "/releases/latest/" not in text


def test_the_mount_is_gated_so_a_silent_failure_cannot_pass_for_a_ready_node():
    text = bootstrap.mounts(((Volume(bucket="a", mount="/a"), Endpoint(url="https://s3.amazonaws.com")),))
    assert f"mountpoint -q {bootstrap.FUSE_ROOT}/a" in text
    assert "exit 1" in text


def test_the_native_path_installs_nothing_and_only_links():
    """RunPod: the host attached the volume, so there is nothing to mount and no FUSE to install."""
    text = bootstrap.symlinks((Volume(bucket="nv-1", mount="/data", prefix="train"),), "/workspace")
    assert "geesefs" not in text
    assert "apt-get" not in text
    assert "mkdir -p /workspace/train" in text
    assert "ln -sfn /workspace/train /data" in text


# --------------------------------------------------------------------------- #
# The phases reach the node's bootstrap
# --------------------------------------------------------------------------- #


def test_the_volume_phase_lands_before_the_plugins_and_inside_the_script():
    phase = bootstrap.mounts(((Volume(bucket="a", mount="/a"), Endpoint(url="https://s3.amazonaws.com")),))
    from skyward.shared.schemas import Image

    text = bootstrap.script(Image(), "skyward", volumes=(phase,))
    assert "phase volumes" in text
    assert text.index("phase skyward") < text.index("phase volumes") < text.index(bootstrap.FOOTER)


def test_a_compute_with_no_volumes_generates_the_script_it_always_did():
    from skyward.shared.schemas import Image

    assert bootstrap.script(Image(), "skyward") == bootstrap.script(Image(), "skyward", (), 1, ())
    assert "phase volumes" not in bootstrap.script(Image(), "skyward")


@pytest.mark.parametrize(
    "volumes",
    [
        ((Volume(bucket="a", mount="/a"), Endpoint(url="https://s3.us-east-1.amazonaws.com")),),
        ((Volume(bucket="a", mount="/a", prefix="p", read_only=False), Endpoint(url="https://gcs", access_key="A'K", secret_key="S K")),),
        (
            (Volume(bucket="a", mount="/a"), Endpoint(url="https://obj", access_key="AK", secret_key="SK", path_style=True)),
            (Volume(bucket="b", mount="/b", read_only=False), Endpoint(url="https://obj", access_key="AK", secret_key="SK", path_style=True)),
        ),
    ],
)
def test_the_mount_phase_is_valid_bash(volumes: tuple[tuple[Volume, Endpoint], ...]):
    """Including when a key contains a quote — the credentials are quoted, not interpolated."""
    from skyward.shared.schemas import Image

    text = bootstrap.script(Image(), "skyward", volumes=(bootstrap.mounts(volumes),))
    bash = shutil.which("bash")
    assert bash, "bash is required for this test"

    result = subprocess.run([bash, "-n"], input=text, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


# --------------------------------------------------------------------------- #
# Who resolves the credentials, decided once at bind
# --------------------------------------------------------------------------- #


async def test_a_provider_that_cannot_mount_refuses_the_compute_rather_than_booting_without_it(
    store: tuple[ComputeStore, BlobStore],
):
    """Silently dropping the volume is a training run that reads an empty directory."""
    computes, blobs = store
    created, _ = await computes.create(ComputeCreate(spec=spec(Volume(bucket="data", mount="/data")), name=None), idempotency_key="k")

    with pytest.raises(CapabilityMismatchError, match="cannot mount data"):
        await Scripted(computes, blobs, Deaf()).bind(created)


async def test_bringing_credentials_for_one_volume_and_not_another_is_refused(
    store: tuple[ComputeStore, BlobStore],
):
    computes, blobs = store
    digest = await blobs.store(b'{"url":"https://r2","access_key":"AK","secret_key":"SK"}')
    mixed = spec(
        Volume(bucket="mine", mount="/mine", storage_sha256=digest),
        Volume(bucket="theirs", mount="/theirs"),
    )
    created, _ = await computes.create(ComputeCreate(spec=mixed, name=None), idempotency_key="k")

    with pytest.raises(CapabilityMismatchError, match="not both"):
        await Scripted(computes, blobs, Attaching()).bind(created)


async def test_credentials_the_client_brought_are_read_from_the_blob_and_never_from_the_spec(
    store: tuple[ComputeStore, BlobStore],
):
    """The escape hatch: an R2 bucket no provider record describes, mounted anyway."""
    computes, blobs = store
    endpoint = Endpoint(url="https://acct.r2.cloudflarestorage.com", access_key="AK", secret_key="SK")

    import msgspec

    digest = await blobs.store(msgspec.json.encode(endpoint))
    created, _ = await computes.create(
        ComputeCreate(spec=spec(Volume(bucket="data", mount="/data", storage_sha256=digest)), name=None),
        idempotency_key="k",
    )

    infrastructure = await Scripted(computes, blobs, Deaf()).bind(created)

    assert len(infrastructure.volumes) == 1
    assert "acct.r2.cloudflarestorage.com" in infrastructure.volumes[0]
    assert "aws_access_key_id = AK" in infrastructure.volumes[0]

    served = await computes.get(created.id)
    assert served.spec.volumes[0].storage_sha256 == digest
    assert "SK" not in msgspec.json.encode(served.spec).decode(), "a secret on the spec is a secret the compute API serves back"


async def test_a_deploy_hint_reaches_the_binding_before_a_machine_is_launched(
    store: tuple[ComputeStore, BlobStore],
):
    """RunPod names its network volume in the launch request, which is over before a node exists."""
    computes, blobs = store
    created, _ = await computes.create(ComputeCreate(spec=spec(Volume(bucket="nv-1", mount="/data")), name=None), idempotency_key="k")

    infrastructure = await Scripted(computes, blobs, Attaching()).bind(created)

    assert infrastructure.binding["attachment"] == "vol-7"
    assert infrastructure.binding["region"] == "us-east-1", "the hint is merged into what initialize built, not put in its place"

    reread = await computes.infrastructure(created.id)
    assert reread.binding["attachment"] == "vol-7", "a daemon that dies after binding must still launch with the volume"
    assert reread.volumes == infrastructure.volumes


async def test_a_compute_with_no_volumes_asks_the_provider_nothing(store: tuple[ComputeStore, BlobStore]):
    computes, blobs = store
    created, _ = await computes.create(ComputeCreate(spec=spec(), name=None), idempotency_key="k")

    infrastructure = await Scripted(computes, blobs, Deaf()).bind(created)

    assert infrastructure.volumes == ()


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #


async def test_aws_reaches_its_buckets_in_the_region_the_compute_was_launched_in():
    provider = AWSProvider("prv_1", "aws", "AK", "SK", None, {})
    mount = await provider.mount({"region": "eu-west-1"}, (Volume(bucket="training", mount="/data"),))

    assert "https://s3.eu-west-1.amazonaws.com" in mount.phases[0]
    assert "--iam --iam-flavor=imdsv1" in mount.phases[0]
    assert mount.binding_patch == {}


async def test_runpod_refuses_a_second_network_volume_rather_than_dropping_it():
    """A pod attaches one; several buckets is a request RunPod cannot satisfy at all."""
    provider = RunPodProvider("prv_1", "runpod", "key", {})
    volumes = (Volume(bucket="nv-1", mount="/a"), Volume(bucket="nv-2", mount="/b"))

    with pytest.raises(CapabilityMismatchError, match="one network volume"):
        await provider.mount({"volume_mount_path": "/workspace"}, volumes)


def test_runpod_is_mountable_and_a_provider_without_the_method_is_not():
    assert isinstance(RunPodProvider("prv_1", "runpod", "key", {}), Mountable)
    assert not isinstance(Deaf(), Mountable)
