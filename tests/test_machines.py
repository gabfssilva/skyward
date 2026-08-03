"""Buying a machine and looking for machines nobody wrote down must not collide.

``create`` launches one machine per node and records its id; ``resolve`` sweeps
the provider for machines no row claims and adopts them. Both write the same
column, and the emitter runs them as separate tasks — so a ``resolve`` that runs
while a ``create`` is between its launch and its record used to see that machine
as an orphan and hand it to a different node, binding one machine to two ranks
and leaking another. That is what turned a four-node broadcast into three
answers and a duplicate.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import msgspec
import pytest

from skyward.providers.jarvislabs import JarvisLabsProvider
from skyward.providers.massed_compute import MassedComputeProvider
from skyward.providers.novita import NovitaProvider
from skyward.providers.runpod import RunPodProvider
from skyward.providers.tensordock import TensorDockProvider
from skyward.providers.vastai import VastAIProvider
from skyward.server.application import machines as machines_module
from skyward.server.application.machines import Machines
from skyward.server.persistence.computes import ComputeStore, Infrastructure
from skyward.server.persistence.db import connect
from skyward.server.persistence.nodes import NodeStore
from skyward.shared.errors import CapabilityMismatchError
from skyward.shared.provider import Binding, Machine, Provider
from skyward.shared.schemas import ComputeCreate, ComputeSpec, Market, NodeBounds, Offer, Options, Page, ProviderRef, Spec

SPEC = ComputeSpec(
    specs=(Spec(provider=ProviderRef(kind="fake"), cpus=1, memory_gb=1),),
    nodes=NodeBounds(desired=4),
)


class ClusterProvider:
    kind = "cluster-provider"

    def __init__(self, allowed: bool) -> None:
        self.allowed = allowed

    def allows_cluster_formation(self, spec: ComputeSpec, offer: Offer) -> bool:
        return self.allowed


@pytest.mark.parametrize(("allowed", "expected"), [(True, True), (False, False)])
def test_unspecified_cluster_mode_follows_provider_capability(allowed: bool, expected: bool) -> None:
    assert machines_module._clustered(ClusterProvider(allowed), SPEC, _offer("region", 1.0)) is expected


def test_explicit_standalone_mode_overrides_provider_capability() -> None:
    spec = msgspec.structs.replace(SPEC, options=Options(cluster=False))
    assert not machines_module._clustered(ClusterProvider(True), spec, _offer("region", 1.0))


def test_explicit_cluster_mode_is_refused_when_the_provider_cannot_form_one() -> None:
    spec = msgspec.structs.replace(SPEC, options=Options(cluster=True))
    with pytest.raises(CapabilityMismatchError, match="does not allow cluster formation"):
        machines_module._clustered(ClusterProvider(False), spec, _offer("region", 1.0))


@pytest.mark.parametrize(
    "adapter",
    [
        JarvisLabsProvider("id", "jarvislabs", "key", {}),
        MassedComputeProvider("id", "massed_compute", "key", {}),
        NovitaProvider("id", "novita", "key", {}),
        TensorDockProvider("id", "tensordock", "token", {}),
        VastAIProvider("id", "vastai", "key", {}),
    ],
)
def test_providers_without_a_shared_network_refuse_cluster_formation(adapter: Provider) -> None:
    assert not adapter.allows_cluster_formation(SPEC, _offer("region", 1.0))


def test_runpod_cluster_formation_depends_on_global_networking() -> None:
    secure = msgspec.structs.replace(_offer("region", 1.0), specific={"cloud_type": "SECURE"})
    community = msgspec.structs.replace(_offer("region", 1.0), specific={"cloud_type": "COMMUNITY"})
    default = RunPodProvider("id", "runpod", "key", {})
    enabled = RunPodProvider("id", "runpod", "key", {"global_networking": True})
    disabled = RunPodProvider("id", "runpod", "key", {"global_networking": False})

    assert not default.allows_cluster_formation(SPEC, secure)
    assert enabled.allows_cluster_formation(SPEC, secure)
    assert not disabled.allows_cluster_formation(SPEC, secure)
    assert not enabled.allows_cluster_formation(SPEC, community)


class FakeProvider:
    """A provider whose ``machines`` reflects what ``launch`` bought.

    ``launch`` makes the machine observable before it lets the caller record it,
    which is the whole window the race lives in: a gate held open there is a
    ``create`` that has spent the money and not yet written the id.
    """

    kind = "fake"

    def __init__(self) -> None:
        self._machines: dict[str, Machine] = {}
        self._counter = 0
        self.launched = asyncio.Event()
        self.proceed = asyncio.Event()

    async def launch(self, binding: Binding, market: Market, count: int, min_count: int) -> tuple[Binding, list[Machine]]:
        machine = Machine(id=f"i-{self._counter}", state="running", host=f"10.0.0.{self._counter}")
        self._counter += 1
        self._machines[machine.id] = machine
        self.launched.set()
        await self.proceed.wait()
        return binding, [machine]

    async def machines(self, binding: Binding) -> dict[str, Machine]:
        return dict(self._machines)


class ScriptedMachines(Machines):
    def __init__(self, computes: ComputeStore, nodes: NodeStore, provider: FakeProvider) -> None:
        super().__init__(computes, nodes, providers=None, offers=None, blobs=None)  # type: ignore[arg-type]
        self._provider = provider

    async def adapter(self, provider_id: str | None) -> FakeProvider:  # type: ignore[override]
        return self._provider


@pytest.fixture
async def wired(tmp_path: Path) -> tuple[ComputeStore, NodeStore, str, FakeProvider]:
    await connect(tmp_path / "skyward.sqlite")
    computes = ComputeStore()
    nodes = NodeStore()

    created, _ = await computes.create(ComputeCreate(spec=SPEC, name=None), idempotency_key="k")
    await computes.bind(
        created.id,
        Infrastructure(provider_id="prv_fake", binding={"net": "n"}, markets=("on_demand",)),
    )

    return computes, nodes, created.id, FakeProvider()


async def test_resolve_does_not_readopt_a_machine_a_launch_is_still_recording(
    wired: tuple[ComputeStore, NodeStore, str, FakeProvider],
) -> None:
    computes, nodes, compute_id, provider = wired
    machines = ScriptedMachines(computes, nodes, provider)

    rows = [await nodes.request(compute_id, generation=1) for _ in range(4)]
    snapshot = await nodes.of(compute_id)

    provider.proceed.clear()
    creating = asyncio.create_task(machines.create(compute_id, rows[1].id))
    await provider.launched.wait()

    resolving = asyncio.create_task(machines.resolve(await computes.get(compute_id), snapshot))
    await asyncio.sleep(0.05)
    provider.proceed.set()

    await asyncio.gather(creating, resolving)

    bound = [node.machine for node in await nodes.of(compute_id) if node.machine]
    assert len(bound) == len(set(bound)), f"a machine got bound to two nodes: {bound}"


async def test_resolve_still_adopts_a_machine_a_dead_process_left_behind(
    wired: tuple[ComputeStore, NodeStore, str, FakeProvider],
) -> None:
    computes, nodes, compute_id, provider = wired
    machines = ScriptedMachines(computes, nodes, provider)

    row = await nodes.request(compute_id, generation=1)
    provider._machines["i-9"] = Machine(id="i-9", state="running", host="10.0.0.9")

    await machines.resolve(await computes.get(compute_id), await nodes.of(compute_id))

    adopted = await nodes.get(compute_id, row.id)
    assert adopted.machine == "i-9", "a machine no row claims must still be picked up"


class FallbackProvider:
    """A provider whose spot launch is always refused and whose on-demand always takes."""

    kind = "fake"

    def __init__(self) -> None:
        self.attempts: list[Market] = []
        self._machines: dict[str, Machine] = {}

    async def launch(self, binding: Binding, market: Market, count: int, min_count: int) -> tuple[Binding, list[Machine]]:
        self.attempts.append(market)
        if market == "spot":
            raise RuntimeError("no spot capacity")
        machine = Machine(id="i-od", state="running", host="10.0.0.1")
        self._machines[machine.id] = machine
        return binding, [machine]

    async def machines(self, binding: Binding) -> dict[str, Machine]:
        return dict(self._machines)


async def test_spot_if_available_falls_back_to_on_demand_when_spot_is_refused(tmp_path: Path) -> None:
    await connect(tmp_path / "skyward.sqlite")
    computes = ComputeStore()
    nodes = NodeStore()

    created, _ = await computes.create(ComputeCreate(spec=SPEC, name=None), idempotency_key="k")
    await computes.bind(
        created.id,
        Infrastructure(provider_id="prv_fake", binding={"net": "n"}, markets=("spot", "on_demand")),
    )

    provider = FallbackProvider()
    machines = ScriptedMachines(computes, nodes, provider)  # type: ignore[arg-type]

    row = await nodes.request(created.id, generation=1)
    await machines.create(created.id, row.id)

    assert provider.attempts == ["spot", "on_demand"], "spot must be tried first, then on-demand"
    node = await nodes.get(created.id, row.id)
    assert node.machine == "i-od", "the node must be placed on the on-demand machine"


class PreemptibleProvider:
    """A provider that warns, ahead of the vanish, which machines it is reclaiming.

    Carries the whole :class:`Preemptible` surface so ``isinstance`` gates it in:
    the catalog attributes and an ``interruptions`` that flags the machines a test
    scripts as being taken back, while ``machines`` still reports them present — the
    proactive window the checker exists to catch.
    """

    kind = "fake"
    credential_fields: tuple[str, ...] = ()
    offers_ttl = timedelta(minutes=1)

    def __init__(self, machines: dict[str, Machine], flagged: dict[str, str]) -> None:
        self._machines = machines
        self._flagged = flagged

    @classmethod
    def create(cls, provider_id: str, name: str, credentials: object, config: object) -> "PreemptibleProvider":
        raise NotImplementedError

    async def offers(self) -> object:
        raise NotImplementedError

    async def machines(self, binding: Binding) -> dict[str, Machine]:
        return dict(self._machines)

    async def interruptions(self, binding: Binding, machine_ids: tuple[str, ...]) -> dict[str, str]:
        return {mid: why for mid, why in self._flagged.items() if mid in machine_ids}


class PreemptibleMachines(Machines):
    def __init__(self, computes: ComputeStore, nodes: NodeStore, provider: PreemptibleProvider) -> None:
        super().__init__(computes, nodes, providers=None, offers=None, blobs=None)  # type: ignore[arg-type]
        self._provider = provider

    async def adapter(self, provider_id: str | None) -> PreemptibleProvider:  # type: ignore[override]
        return self._provider


async def test_resolve_marks_a_warned_node_lost_while_the_machine_is_still_present(tmp_path: Path) -> None:
    from skyward.shared.provider import Preemptible

    await connect(tmp_path / "skyward.sqlite")
    computes = ComputeStore()
    nodes = NodeStore()

    created, _ = await computes.create(ComputeCreate(spec=SPEC, name=None), idempotency_key="k")
    await computes.bind(
        created.id,
        Infrastructure(provider_id="prv_fake", binding={"net": "n"}, markets=("spot",)),
    )

    machine = Machine(id="i-spot", state="running", host="10.0.0.5")
    provider = PreemptibleProvider({"i-spot": machine}, flagged={"i-spot": "spot-interruption"})
    assert isinstance(provider, Preemptible), "the fake must satisfy the protocol resolve gates on"

    machines = PreemptibleMachines(computes, nodes, provider)

    row = await nodes.request(created.id, generation=1)
    await nodes.reachable(row.id, machine)

    await machines.resolve(await computes.get(created.id), await nodes.of(created.id))

    lost = await nodes.get(created.id, row.id)
    assert lost.state == "lost", "a machine the provider warns it is reclaiming must become a deficit"
    assert lost.last_error is not None and lost.last_error.message == "spot-interruption"


def _offer(region: str, price: float) -> Offer:
    now = datetime.now(UTC)
    return Offer(
        id=f"aws-{region}-x",
        provider_id="prv_fake",
        provider_name="fake",
        kind="fake",
        instance_type="x",
        accelerator_count=0,
        cpus=1,
        memory_gb=1,
        fetched_at=now,
        expires_at=now,
        region=region,
        spot_price=price,
        on_demand_price=price * 2,
    )


class RegionProvider:
    """A provider that sells only in one region and refuses every other's markets.

    ``initialize`` mints a region-scoped binding and records the region it set up in;
    ``launch`` refuses unless the binding is the one region with quota; ``release``
    records the region handed back. Together they let a test watch a placement walk
    from the cheapest region to the one that will sell, cleaning up as it goes.
    """

    kind = "fake"

    def __init__(self, sells_in: str, *, allows_cluster: bool = True) -> None:
        self.sells_in = sells_in
        self.allows_cluster = allows_cluster
        self.initialized: list[str] = []
        self.initialized_cluster: list[bool | None] = []
        self.released: list[str] = []
        self._machines: dict[str, Machine] = {}

    def allows_cluster_formation(self, spec: ComputeSpec, offer: Offer) -> bool:
        return self.allows_cluster

    async def initialize(self, compute_id: str, spec: ComputeSpec, offer: Offer, market: Market, public_key: str) -> Binding:
        assert offer.region is not None
        self.initialized.append(offer.region)
        self.initialized_cluster.append(spec.options.cluster)
        return {"region": offer.region}

    async def launch(self, binding: Binding, market: Market, count: int, min_count: int) -> tuple[Binding, list[Machine]]:
        if binding["region"] != self.sells_in:
            raise RuntimeError(f"no quota in {binding['region']}")
        machine = Machine(id=f"i-{binding['region']}", state="running", host="10.0.0.1")
        self._machines[machine.id] = machine
        return binding, [machine]

    async def release(self, binding: Binding) -> None:
        self.released.append(binding["region"])

    async def machines(self, binding: Binding) -> dict[str, Machine]:
        return dict(self._machines)


class FakeOffers:
    def __init__(self, offers: tuple[Offer, ...]) -> None:
        self._offers = offers

    async def list(self, **_: object) -> Page[Offer]:
        return Page(items=self._offers)


class RegionMachines(Machines):
    def __init__(self, computes: ComputeStore, nodes: NodeStore, provider: RegionProvider, offers: FakeOffers) -> None:
        super().__init__(computes, nodes, providers=None, offers=offers, blobs=None)  # type: ignore[arg-type]
        self._provider = provider

    async def adapter(self, provider_id: str | None) -> RegionProvider:  # type: ignore[override]
        return self._provider


async def test_placement_relocates_to_a_region_that_will_sell_and_releases_the_one_that_refused(
    tmp_path: Path,
) -> None:
    await connect(tmp_path / "skyward.sqlite")
    computes = ComputeStore()
    nodes = NodeStore()

    created, _ = await computes.create(ComputeCreate(spec=SPEC, name=None), idempotency_key="k")

    cheap, quota = _offer("cheap", 0.10), _offer("quota", 0.20)
    provider = RegionProvider(sells_in="quota")
    machines = RegionMachines(computes, nodes, provider, FakeOffers((cheap, quota)))

    row = await nodes.request(created.id, generation=1)
    await machines.create(created.id, row.id)

    node = await nodes.get(created.id, row.id)
    assert node.machine == "i-quota", "the node must land in the region that had capacity"

    assert provider.initialized == ["cheap", "quota"], "the cheapest is bound first, then the one that sells"
    assert provider.released == ["cheap"], "the region that refused must be released, not left to leak"

    infrastructure = await computes.infrastructure(created.id)
    assert infrastructure.offer_id == quota.id, "the compute must be re-bound to the region that sold"
    assert infrastructure.binding["region"] == "quota"
    assert infrastructure.binding["skyward_cluster"] is True


async def test_explicit_standalone_mode_is_persisted_with_the_provider_binding(tmp_path: Path) -> None:
    await connect(tmp_path / "skyward.sqlite")
    computes = ComputeStore()
    nodes = NodeStore()
    spec = msgspec.structs.replace(SPEC, options=Options(cluster=False))
    created, _ = await computes.create(ComputeCreate(spec=spec, name=None), idempotency_key="k")
    offer = _offer("only", 0.10)
    provider = RegionProvider(sells_in="only")
    machines = RegionMachines(computes, nodes, provider, FakeOffers((offer,)))

    row = await nodes.request(created.id, generation=1)
    await machines.create(created.id, row.id)

    assert (await computes.infrastructure(created.id)).binding["skyward_cluster"] is False
    assert provider.initialized_cluster == [False]


async def test_provider_default_is_resolved_before_initialize(tmp_path: Path) -> None:
    await connect(tmp_path / "skyward.sqlite")
    computes = ComputeStore()
    nodes = NodeStore()
    created, _ = await computes.create(ComputeCreate(spec=SPEC, name=None), idempotency_key="k")
    offer = _offer("only", 0.10)
    provider = RegionProvider(sells_in="only", allows_cluster=False)
    machines = RegionMachines(computes, nodes, provider, FakeOffers((offer,)))

    row = await nodes.request(created.id, generation=1)
    await machines.create(created.id, row.id)

    assert provider.initialized_cluster == [False]
    assert (await computes.infrastructure(created.id)).binding["skyward_cluster"] is False


async def test_a_second_node_inherits_the_region_the_first_settled_on(tmp_path: Path) -> None:
    await connect(tmp_path / "skyward.sqlite")
    computes = ComputeStore()
    nodes = NodeStore()

    created, _ = await computes.create(ComputeCreate(spec=SPEC, name=None), idempotency_key="k")

    cheap, quota = _offer("cheap", 0.10), _offer("quota", 0.20)
    provider = RegionProvider(sells_in="quota")
    machines = RegionMachines(computes, nodes, provider, FakeOffers((cheap, quota)))

    first, second = [await nodes.request(created.id, generation=1) for _ in range(2)]
    await machines.create(created.id, first.id)
    await machines.create(created.id, second.id)

    assert provider.initialized == ["cheap", "quota"], "the region hunt happens once; the second node inherits it"
    assert provider.released == ["cheap"], "the abandoned region is released once, not per node"
    assert (await nodes.get(created.id, second.id)).machine == "i-quota"
