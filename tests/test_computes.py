"""What the daemon refuses when a compute is asked for, and how it says so.

Both of these were a `500` before: an integrity error and a `RuntimeError` on
their way out through the exception handler. A caller can act on a refusal that
names itself — pick another name, ask the daemon that is holding the compute —
and can do nothing at all with an internal error.
"""

import asyncio
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import msgspec
import pytest

from skyward.server.application.machines import Machines
from skyward.server.application.mock import SPEC
from skyward.server.application.node import Node as ApplicationNode
from skyward.server.application.runtimes import Files, Runtime, Runtimes
from skyward.server.application.source import Source
from skyward.server.persistence.computes import ComputeStore, Infrastructure
from skyward.server.persistence.db import connect
from skyward.server.persistence.events import EventStore
from skyward.server.persistence.functions import BlobStore
from skyward.server.persistence.nodes import NodeStore
from skyward.server.persistence.offers import OfferCache
from skyward.server.persistence.providers import ProviderStore
from skyward.server.persistence.tables import ComputeRow
from skyward.shared.errors import ComputeNotConnectedError, NameTakenError
from skyward.shared.provider import Machine
from skyward.shared.schemas import Compute, ComputeCreate, ComputeStatus, Image, Node

pytestmark = pytest.mark.local


def describe_naming_a_compute() -> None:
    async def it_is_refused_when_another_compute_already_has_the_name(tmp_path: Path) -> None:
        await connect(tmp_path / "skyward.sqlite")
        store = ComputeStore()
        first, _ = await store.create(ComputeCreate(spec=SPEC, name="training"), idempotency_key="first")

        with pytest.raises(NameTakenError) as refused:
            await store.create(ComputeCreate(spec=SPEC, name="training"), idempotency_key="second")

        assert refused.value.details["compute"] == first.id
        assert refused.value.status == 409

    async def it_lets_two_computes_go_unnamed(tmp_path: Path) -> None:
        await connect(tmp_path / "skyward.sqlite")
        store = ComputeStore()

        first, _ = await store.create(ComputeCreate(spec=SPEC), idempotency_key="first")
        second, _ = await store.create(ComputeCreate(spec=SPEC), idempotency_key="second")

        assert first.id != second.id, "a name nobody gave is not a name two computes share"


def describe_listing_computes() -> None:
    async def they_come_newest_first(tmp_path: Path) -> None:
        store = await _store(tmp_path)
        for index in range(3):
            await store.create(ComputeCreate(spec=SPEC, name=f"c{index}"), idempotency_key=f"k{index}")

        page = await store.list(None, 50, None, None, None)

        assert [compute.name for compute in page.items] == ["c2", "c1", "c0"]

    async def the_live_ones_are_asked_for_apart_from_the_finished(tmp_path: Path) -> None:
        store = await _store(tmp_path)
        running, _ = await store.create(ComputeCreate(spec=SPEC, name="running"), idempotency_key="k1")
        gone, _ = await store.create(ComputeCreate(spec=SPEC, name="gone"), idempotency_key="k2")
        await store.observe(gone.id, ComputeStatus(state="deleted", observed_generation=1, nodes_ready=0, nodes_total=0))

        live = await store.list(None, 50, None, None, True)
        finished = await store.list(None, 50, None, None, False)

        assert [compute.name for compute in live.items] == ["running"]
        assert [compute.name for compute in finished.items] == ["gone"]

    async def a_page_picks_up_below_the_one_before_it(tmp_path: Path) -> None:
        store = await _store(tmp_path)
        for index in range(4):
            await store.create(ComputeCreate(spec=SPEC, name=f"c{index}"), idempotency_key=f"k{index}")

        first = await store.list(None, 2, None, None, None)
        second = await store.list(first.next_cursor, 2, None, None, None)

        assert [compute.name for compute in first.items] == ["c3", "c2"]
        assert [compute.name for compute in second.items] == ["c1", "c0"]


def describe_reaching_a_compute_this_daemon_is_not_holding() -> None:
    async def it_is_told_rather_than_raised_through(tmp_path: Path) -> None:
        files = Files(Runtimes(listener=lambda *_: None, output=lambda *_: None, sample=lambda *_: None, phase=lambda *_: None))

        with pytest.raises(ComputeNotConnectedError) as refused:
            await files.run("cmp_elsewhere", "all", "echo hello")

        assert refused.value.status == 409
        assert refused.value.retryable, "another daemon holds it, or this one has not picked it up yet"


def describe_a_machine_that_is_bought_and_never_says_where_it_is() -> None:
    async def it_is_given_up_on_once_the_window_closes(tmp_path: Path) -> None:
        machines, compute, node = await _bought(tmp_path / "skyward.sqlite", provision_timeout=0.01)
        await asyncio.sleep(0.05)

        await machines.resolve(compute, await NodeStore().of(compute.id))

        given_up = await NodeStore().get(compute.id, node.id)
        assert given_up.state == "lost"
        assert given_up.last_error is not None
        assert "never published an address" in given_up.last_error.message

    async def it_is_waited_on_while_the_window_is_open(tmp_path: Path) -> None:
        machines, compute, node = await _bought(tmp_path / "skyward.sqlite", provision_timeout=600.0)

        await machines.resolve(compute, await NodeStore().of(compute.id))

        assert (await NodeStore().get(compute.id, node.id)).state == "provisioning", "a machine still coming up is not a machine lost"


def describe_a_machine_the_provider_says_is_still_getting_closer() -> None:
    async def it_is_waited_on_for_as_long_as_it_keeps_moving(tmp_path: Path) -> None:
        machines, compute, node = await _bought(
            tmp_path / "skyward.sqlite",
            provision_timeout=0.01,
            reported=(("downloading", 0.10), ("downloading", 0.55), ("downloading", 0.90)),
        )

        for _ in range(3):
            await asyncio.sleep(0.05)
            await machines.resolve(compute, await NodeStore().of(compute.id))

        waited = await NodeStore().get(compute.id, node.id)
        assert waited.state == "provisioning", "a machine pulling its image is a machine still coming up"

    async def it_is_given_up_on_once_it_stops_moving(tmp_path: Path) -> None:
        machines, compute, node = await _bought(
            tmp_path / "skyward.sqlite",
            provision_timeout=0.01,
            reported=(("downloading", 0.55), ("downloading", 0.55)),
        )

        for _ in range(2):
            await asyncio.sleep(0.05)
            await machines.resolve(compute, await NodeStore().of(compute.id))

        given_up = await NodeStore().get(compute.id, node.id)
        assert given_up.state == "lost"
        assert given_up.last_error is not None
        assert "downloading (55%)" in given_up.last_error.message, "the reason names what the machine was doing"


def describe_binding_a_compute() -> None:
    async def the_first_key_written_is_the_key_kept(tmp_path: Path) -> None:
        """Two daemons on one file race their minted pairs; the machines trust the winner's."""
        store = await _store(tmp_path)
        compute, _ = await store.create(ComputeCreate(spec=SPEC), idempotency_key="raced")

        await store.bind(compute.id, Infrastructure(provider_id="prv_1", binding={"a": 1}, private_key="winner"))
        await store.bind(compute.id, Infrastructure(provider_id="prv_1", binding={"b": 2}, private_key="loser"))

        stored = await store.infrastructure(compute.id)
        assert stored.private_key == "winner"
        assert stored.binding == {"a": 1}, "a binding the fleet was launched under is not overwritten"

    async def a_rebind_carrying_the_same_key_lands(tmp_path: Path) -> None:
        """Relocation binds the compute into another region under the key it already has."""
        store = await _store(tmp_path)
        compute, _ = await store.create(ComputeCreate(spec=SPEC), idempotency_key="moved")

        await store.bind(compute.id, Infrastructure(provider_id="prv_1", binding={"region": "us"}, private_key="key"))
        await store.bind(compute.id, Infrastructure(provider_id="prv_1", binding={"region": "eu"}, private_key="key"))

        assert (await store.infrastructure(compute.id)).binding == {"region": "eu"}


def describe_taking_hold_of_one_machine() -> None:
    def it_is_claimed_by_the_first_connect_in_flight() -> None:
        runtime = _runtime()

        assert runtime.claim("nod_1")
        assert not runtime.claim("nod_1"), "a second offer mid-flight must not hold a second channel"

    def it_can_be_offered_again_after_a_failed_connect() -> None:
        runtime = _runtime()
        runtime.claim("nod_1")

        runtime.release("nod_1")

        assert runtime.claim("nod_1")

    def it_stays_held_once_the_node_is_built() -> None:
        runtime = _runtime()
        runtime.claim("nod_1")
        runtime.track("nod_1", _node())

        runtime.release("nod_1")

        assert not runtime.claim("nod_1"), "membership is what refuses the claim once the node exists"


def describe_two_nodes_behind_one_address() -> None:
    """Marketplace machines NAT-share a public IP, so an advertised address does not name a node."""

    def it_routes_each_standalone_client_through_its_own_tunnel() -> None:
        runtime = _runtime(cluster=False)
        first, second = _node(host="140.82.47.249"), _node(host="140.82.47.249")
        first.tunnel, second.tunnel = 40001, 40002
        runtime.track("nod_1", first)
        runtime.track("nod_2", second)

        assert first.seed == second.seed, "the collision under test — both nodes advertise the same address"
        assert runtime.address_map("nod_1")(first.seed) == "127.0.0.1:40001"
        assert runtime.address_map("nod_2")(second.seed) == "127.0.0.1:40002"

    def it_reads_the_tunnel_live_across_a_reconnect() -> None:
        runtime = _runtime(cluster=False)
        node = _node()
        node.tunnel = 40001
        runtime.track("nod_1", node)
        via = runtime.address_map("nod_1")

        node.tunnel = 40002

        assert via(node.seed) == "127.0.0.1:40002"

    async def it_refuses_to_cluster_nodes_it_cannot_tell_apart() -> None:
        runtime = _runtime()
        first, second = _node(host="140.82.47.249"), _node(host="140.82.47.249")
        first.tunnel, second.tunnel = 40001, 40002
        runtime.track("nod_1", first)
        runtime.track("nod_2", second)

        with pytest.raises(RuntimeError, match="sharing an address"):
            await runtime.system()


def describe_a_spec_written_under_the_old_vocabulary() -> None:
    async def it_is_mended_on_the_next_open(tmp_path: Path) -> None:
        """A row from before ``NodeBounds.desired`` became ``initial`` must still decode."""
        database = tmp_path / "skyward.sqlite"
        await connect(database)
        store = ComputeStore()
        compute, _ = await store.create(ComputeCreate(spec=SPEC), idempotency_key="aged")
        await ComputeRow.raw(
            "UPDATE computes SET spec = json_remove(json_set(spec, '$.nodes.desired', "
            "json_extract(spec, '$.nodes.initial')), '$.nodes.initial')"
        ).run()

        await connect(database)

        mended = await ComputeStore().get(compute.id)
        assert mended.spec.nodes == SPEC.nodes


async def _bought(
    database: Path,
    provision_timeout: float,
    reported: tuple[tuple[str, float | None], ...] = (),
) -> tuple[Machines, Compute, Node]:
    """A compute whose one node has a machine the provider reports without an address.

    ``reported`` is what the provider says the machine is doing and how far into it,
    one entry per listing; the last one is repeated once they run out.
    """
    await connect(database)
    computes, nodes, providers = ComputeStore(), NodeStore(), ProviderStore()
    spec = msgspec.structs.replace(SPEC, options=msgspec.structs.replace(SPEC.options, provision_timeout=provision_timeout))
    compute, _ = await computes.create(ComputeCreate(spec=spec), idempotency_key="bought")
    await computes.bind(compute.id, Infrastructure(provider_id="prv_1", binding={"prefix": "skyward-"}))

    node = await nodes.request(compute.id, compute.generation)
    await nodes.launched(node.id, Machine(id="m1", state="running"))

    machines = Machines(
        computes=computes,
        nodes=nodes,
        providers=providers,
        offers=OfferCache(providers),
        blobs=BlobStore(),
        events=EventStore(),
    )
    progress = deque(reported)
    machines.adapter = lambda _: _answering(progress)
    return machines, await computes.get(compute.id), node


async def _answering(progress: deque[tuple[str, float | None]]) -> Any:
    """A provider with one machine that has no address, and is asked nothing else."""
    return SimpleNamespace(machines=lambda _: _listing(progress))


async def _listing(progress: deque[tuple[str, float | None]]) -> dict[str, Machine]:
    seen = progress.popleft() if len(progress) > 1 else next(iter(progress), (None, None))
    return {"m1": Machine(id="m1", state="running", progress=seen[0], completion=seen[1])}


async def _store(tmp_path: Path) -> ComputeStore:
    await connect(tmp_path / "skyward.sqlite")
    return ComputeStore()


def _runtime(cluster: bool = True) -> Runtime:
    return Runtime("cmp_1", Source(arguments=("skyward",)), private_key="key", cluster=cluster)


def _node(host: str = "127.0.0.1") -> ApplicationNode:
    quiet = lambda *args: None  # noqa: E731
    return ApplicationNode(
        Machine(id="m1", state="running", host=host),
        compute="cmp_1",
        private_key="key",
        image=Image(),
        source=Source(arguments=("skyward",)),
        listener=quiet,
        output=quiet,
        sample=quiet,
        phase=quiet,
    )
