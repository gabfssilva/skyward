"""An image is worth reusing only for as long as its name is still true.

Bootstrap is the same apt packages, the same wheels and the same driver every time,
and a provider that can snapshot a running machine can be asked to keep that work.
What makes it safe is that the snapshot is named after what went into it, so a
compute booting from one is booting from its own environment and not from something
that merely used to be it.

Which is the whole reason a local skyward refuses to bake: it is installed from a
wheel built out of the checkout, and those bytes change with every edit that has been
committed nowhere. The name would go on claiming an environment that no longer
exists, and tomorrow's compute would boot into yesterday's code without a word.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from skyward.application.machines import Machines
from skyward.application.provider import Bakeable, Binding, Machine
from skyward.application.reconciler import Reconciler, Wakeup
from skyward.persistence.computes import ComputeStore, GenerationStore, Infrastructure
from skyward.persistence.db import connect
from skyward.persistence.events import EventStore
from skyward.persistence.functions import BlobStore
from skyward.persistence.nodes import NodeStore
from skyward.persistence.tasks import TaskStore
from skyward.protocol.schemas import (
    ComputeCreate,
    ComputeSpec,
    Image,
    Market,
    MetricSpec,
    NodeBounds,
    Offer,
    Page,
    PipIndex,
    ProviderRef,
    Spec,
)

SOURCE = "skyward"
WARM = Image(warm=True, skyward="pypi")


def spec(image: Image) -> ComputeSpec:
    return ComputeSpec(
        specs=(Spec(provider=ProviderRef(kind="fake"), cpus=1, memory_gb=1),),
        nodes=NodeBounds(desired=2),
        image=image,
    )


def offer() -> Offer:
    stamp = datetime.now(UTC)
    return Offer(
        id="fake-cpu",
        provider_id="prv_fake",
        provider_name="fake",
        kind="fake",
        instance_type="fake.cpu",
        accelerator_count=0,
        cpus=1,
        memory_gb=1,
        region="fake-1",
        on_demand_price=0.10,
        fetched_at=stamp,
        expires_at=stamp,
    )


class WarmProvider:
    """A provider that remembers what it was asked to keep, and hands it back.

    ``initialize`` resolves a boot image the way every adapter does — into
    ``binding["image"]`` — so a test can watch that one value become the warm one and
    nothing else change. ``existing`` is what the provider already holds before the
    test starts, which is the difference between a cold environment and one somebody
    else already baked.

    A commit does not make the image visible to ``baked``: what has just been created
    is still registering, and cannot be booted from. That window is real on every
    provider, and it is precisely when the second ready tick arrives — so it is what a
    guard against baking twice has to hold against.
    """

    kind = "fake"

    def __init__(self, existing: str | None = None) -> None:
        self.existing = existing
        self.committed: list[tuple[str, str]] = []
        self.asked: list[str] = []

    async def initialize(self, compute_id: str, spec: ComputeSpec, offer: Offer, market: Market, public_key: str) -> Binding:
        return {"region": "fake-1", "image": "cold-image"}

    async def launch(self, binding: Binding, market: Market, count: int, min_count: int) -> tuple[Binding, list[Machine]]:
        return binding, [Machine(id="i-0", state="running", host="10.0.0.1")]

    async def machines(self, binding: Binding) -> dict[str, Machine]:
        return {}

    async def bake(self, binding: Binding, machine_id: str, tag: str) -> str:
        self.committed.append((machine_id, tag))
        return f"warm-{tag}"

    async def baked(self, binding: Binding, tag: str) -> str | None:
        self.asked.append(tag)
        return self.existing


class ColdProvider:
    """A provider with nowhere to put a snapshot — the shape of RunPod, Novita, Vast."""

    kind = "fake"

    async def initialize(self, compute_id: str, spec: ComputeSpec, offer: Offer, market: Market, public_key: str) -> Binding:
        return {"region": "fake-1", "image": "cold-image"}

    async def machines(self, binding: Binding) -> dict[str, Machine]:
        return {}


class RefusingProvider(WarmProvider):
    """A provider whose commit always fails — a full disk, a revoked permission."""

    async def bake(self, binding: Binding, machine_id: str, tag: str) -> str:
        raise RuntimeError("no space left on device")


class FakeOffers:
    def __init__(self, offers: tuple[Offer, ...]) -> None:
        self._offers = offers

    async def list(self, **_: object) -> Page[Offer]:
        return Page(items=self._offers)


class WarmMachines(Machines):
    def __init__(self, computes: ComputeStore, nodes: NodeStore, provider: object, offers: FakeOffers) -> None:
        super().__init__(computes, nodes, providers=None, offers=offers, blobs=None)  # type: ignore[arg-type]
        self._provider = provider

    async def adapter(self, provider_id: str | None) -> object:  # type: ignore[override]
        return self._provider


@pytest.fixture
async def wired(tmp_path: Path) -> tuple[ComputeStore, NodeStore, FakeOffers]:
    await connect(tmp_path / "skyward.sqlite")
    return ComputeStore(), NodeStore(), FakeOffers((offer(),))


async def compute_with(computes: ComputeStore, image: Image, bound: bool = True) -> str:
    created, _ = await computes.create(ComputeCreate(spec=spec(image), name=None), idempotency_key="k")
    if bound:
        await computes.bind(
            created.id,
            Infrastructure(provider_id="prv_fake", binding={"region": "fake-1", "image": "cold-image"}, markets=("on_demand",)),
        )
    return created.id


def test_the_same_environment_always_names_the_same_image():
    left = Image(base="ubuntu:24.04", python="3.13", pip=("torch",))
    right = Image(base="ubuntu:24.04", python="3.13", pip=("torch",))

    assert left.content_hash(SOURCE) == right.content_hash(SOURCE)
    assert len(left.content_hash(SOURCE)) == 12


@pytest.mark.parametrize(
    ("image", "source"),
    [
        (Image(base="ubuntu:22.04"), SOURCE),
        (Image(python="3.12"), SOURCE),
        (Image(pip=("torch",)), SOURCE),
        (Image(apt=("git",)), SOURCE),
        (Image(pip_indexes=(PipIndex(url="https://example.test"),)), SOURCE),
        (Image(), "git+https://github.com/gabfssilva/skyward.git"),
    ],
)
def test_everything_the_bootstrap_installs_moves_the_name(image: Image, source: str):
    assert image.content_hash(source) != Image().content_hash(SOURCE)


@pytest.mark.parametrize(
    "image",
    [
        Image(env={"HF_TOKEN": "x"}),
        Image(shell_vars={"LD_LIBRARY_PATH": "/usr/local/cuda"}),
        Image(metrics=(MetricSpec(name="gpu", command="nvidia-smi", interval=5.0),)),
        Image(includes_sha256="a" * 64),
        Image(bootstrap_timeout=60),
        Image(warm=True),
    ],
)
def test_what_the_bootstrap_redoes_on_every_boot_leaves_the_name_alone(image: Image):
    assert image.content_hash(SOURCE) == Image().content_hash(SOURCE)


async def test_a_compute_running_a_local_skyward_bakes_nothing(
    wired: tuple[ComputeStore, NodeStore, FakeOffers],
) -> None:
    computes, nodes, offers = wired
    provider = WarmProvider()
    machines = WarmMachines(computes, nodes, provider, offers)

    compute_id = await compute_with(computes, Image(warm=True, skyward="local"))
    row = await nodes.request(compute_id, generation=1)
    await nodes.launched(row.id, Machine(id="i-0", state="running", host="10.0.0.1"))

    await machines.bake(compute_id, row.id)

    assert provider.committed == [], "a wheel whose bytes change on every edit must not name an image"
    assert provider.asked == [], "the refusal is ours; the provider is never given the chance to answer"


async def test_an_image_that_did_not_ask_to_be_kept_bakes_nothing(
    wired: tuple[ComputeStore, NodeStore, FakeOffers],
) -> None:
    computes, nodes, offers = wired
    provider = WarmProvider()
    machines = WarmMachines(computes, nodes, provider, offers)

    compute_id = await compute_with(computes, Image(skyward="pypi"))
    row = await nodes.request(compute_id, generation=1)
    await nodes.launched(row.id, Machine(id="i-0", state="running", host="10.0.0.1"))

    await machines.bake(compute_id, row.id)

    assert provider.committed == [], "a snapshot nobody asked for is storage nobody agreed to pay for"
    assert provider.asked == [], "a compute that is not baking must not even ask"


async def test_a_node_that_is_not_rank_zero_is_never_the_one_kept(
    wired: tuple[ComputeStore, NodeStore, FakeOffers],
) -> None:
    computes, nodes, offers = wired
    provider = WarmProvider()
    machines = WarmMachines(computes, nodes, provider, offers)

    compute_id = await compute_with(computes, WARM)
    first, second = [await nodes.request(compute_id, generation=1) for _ in range(2)]
    await nodes.launched(first.id, Machine(id="i-0", state="running", host="10.0.0.1"))
    await nodes.launched(second.id, Machine(id="i-1", state="running", host="10.0.0.2"))

    await machines.bake(compute_id, second.id)
    assert provider.committed == [], "one snapshot describes the whole compute, and it is rank zero's"

    await machines.bake(compute_id, first.id)
    assert [machine for machine, _ in provider.committed] == ["i-0"]


async def test_a_node_that_serves_twice_is_kept_once(
    wired: tuple[ComputeStore, NodeStore, FakeOffers],
) -> None:
    computes, nodes, offers = wired
    provider = WarmProvider()
    machines = WarmMachines(computes, nodes, provider, offers)

    compute_id = await compute_with(computes, WARM)
    row = await nodes.request(compute_id, generation=1)
    await nodes.launched(row.id, Machine(id="i-0", state="running", host="10.0.0.1"))

    await machines.bake(compute_id, row.id)
    await machines.bake(compute_id, row.id)

    assert len(provider.committed) == 1, "a node re-offered by a later tick must not be snapshotted again"


async def test_an_environment_the_provider_already_holds_is_not_baked_over(
    wired: tuple[ComputeStore, NodeStore, FakeOffers],
) -> None:
    computes, nodes, offers = wired
    provider = WarmProvider(existing="warm-already")
    machines = WarmMachines(computes, nodes, provider, offers)

    compute_id = await compute_with(computes, WARM)
    row = await nodes.request(compute_id, generation=1)
    await nodes.launched(row.id, Machine(id="i-0", state="running", host="10.0.0.1"))

    await machines.bake(compute_id, row.id)

    assert provider.asked, "the provider is the authority on what it already has"
    assert provider.committed == [], "a second image of the same environment is storage billed twice"


async def test_a_provider_that_will_not_commit_does_not_take_the_node_down_with_it(
    wired: tuple[ComputeStore, NodeStore, FakeOffers],
) -> None:
    computes, nodes, offers = wired
    machines = WarmMachines(computes, nodes, RefusingProvider(), offers)

    compute_id = await compute_with(computes, WARM)
    row = await nodes.request(compute_id, generation=1)
    await nodes.launched(row.id, Machine(id="i-0", state="running", host="10.0.0.1"))

    await machines.bake(compute_id, row.id)

    assert (await nodes.get(compute_id, row.id)).machine == "i-0", "nothing about baking is on the path to running work"


async def test_a_bind_boots_from_the_image_the_environment_was_baked_into(
    wired: tuple[ComputeStore, NodeStore, FakeOffers],
) -> None:
    computes, nodes, offers = wired
    provider = WarmProvider(existing="warm-abc123")
    machines = WarmMachines(computes, nodes, provider, offers)

    compute_id = await compute_with(computes, WARM, bound=False)
    infrastructure = await machines.bind(await computes.get(compute_id))

    assert infrastructure.binding["image"] == "warm-abc123", "a warm image is another value under the key the adapter already writes"
    assert provider.asked == [WARM.content_hash(SOURCE)], "the environment is what is asked for, not the compute"


async def test_a_bind_keeps_the_adapters_own_image_when_nothing_was_baked(
    wired: tuple[ComputeStore, NodeStore, FakeOffers],
) -> None:
    computes, nodes, offers = wired
    provider = WarmProvider()
    machines = WarmMachines(computes, nodes, provider, offers)

    compute_id = await compute_with(computes, WARM, bound=False)
    infrastructure = await machines.bind(await computes.get(compute_id))

    assert infrastructure.binding["image"] == "cold-image", "no image yet is a cold boot, not a failure"


async def test_the_node_reporting_itself_ready_is_what_sets_the_keeping_off(
    wired: tuple[ComputeStore, NodeStore, FakeOffers],
) -> None:
    computes, nodes, offers = wired
    provider = WarmProvider()
    machines = WarmMachines(computes, nodes, provider, offers)
    reconciler = Reconciler(
        computes=computes,
        generations=GenerationStore(computes),
        nodes=nodes,
        tasks=TaskStore(computes, nodes, BlobStore()),
        machines=machines,
        events=EventStore(),
        wake=Wakeup(),
    )

    compute_id = await compute_with(computes, WARM)
    row = await nodes.request(compute_id, generation=1)
    await nodes.launched(row.id, Machine(id="i-0", state="running", host="10.0.0.1"))

    await reconciler.observed(compute_id, row.id, "ready", "")

    assert [machine for machine, _ in provider.committed] == ["i-0"], "a machine is worth an image only once it is known to be built"


async def test_a_provider_that_cannot_snapshot_is_never_asked_to(
    wired: tuple[ComputeStore, NodeStore, FakeOffers],
) -> None:
    computes, nodes, offers = wired
    provider = ColdProvider()
    machines = WarmMachines(computes, nodes, provider, offers)

    assert not isinstance(provider, Bakeable), "the gate is the two methods and nothing else"

    compute_id = await compute_with(computes, WARM, bound=False)
    infrastructure = await machines.bind(await computes.get(compute_id))

    assert infrastructure.binding["image"] == "cold-image"

    row = await nodes.request(compute_id, generation=1)
    await nodes.launched(row.id, Machine(id="i-0", state="running", host="10.0.0.1"))
    await machines.bake(compute_id, row.id)
