"""How the workers of one compute find each other, and how the daemon finds them.

The workers form one casty cluster and the daemon is a client of it. The only
decision in that is which node opens the cluster rather than joining it, and the
rows cannot answer it: a machine gets its address when the provider hands it over,
not in rank order, so the lowest-ranked node with an address a second from now is
not the one it is right now. Deciding it twice is a compute with two clusters, each
holding some of the workers and neither reachable from the other — with the daemon
inside whichever one it dialled first.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import casty
import msgspec
import pytest

import skyward.server.application.runtimes as runtimes_module
from skyward.server.application.connector import Connector, _peers, _seeds
from skyward.server.application.mock import OFFER, SPEC
from skyward.server.application.node import Node as Machinery
from skyward.server.application.runtimes import Runtime, Runtimes
from skyward.server.application.source import Source
from skyward.server.persistence.computes import ComputeStore, Infrastructure
from skyward.server.persistence.db import connect as database
from skyward.server.persistence.events import EventStore
from skyward.server.persistence.functions import BlobStore
from skyward.server.persistence.nodes import NodeStore
from skyward.shared.errors import ComputeNotConnectedError
from skyward.shared.provider import Machine
from skyward.shared.schemas import ComputeCreate, Image, Node, NodeState
from skyward.worker import worker

pytestmark = pytest.mark.local


def describe_which_node_opens_the_cluster() -> None:
    def it_is_the_first_machine_the_daemon_takes_hold_of() -> None:
        runtime = _runtime()

        assert runtime.opens("nod_2", formed=False)

    def it_is_exactly_one_of_them_when_every_machine_connects_at_once() -> None:
        """A compute opens all of its machines together, and the answer is a door, not none.

        Asking whether this is the only machine here answers no for every one of
        fifteen connects a moment apart — each of the other fourteen has already
        claimed — and a compute where nobody opens the cluster is fifteen workers
        waiting out the seed timeout, dying and being started again.
        """
        runtime = _runtime()
        machines = tuple(f"nod_{rank}" for rank in range(15))
        for machine in machines:
            runtime.claim(machine)

        assert sum(runtime.opens(machine, formed=False) for machine in machines) == 1

    def it_is_still_the_same_machine_when_its_own_connect_asks_again() -> None:
        runtime = _runtime()
        runtime.claim("nod_2")

        assert runtime.opens("nod_2", formed=False)
        assert runtime.opens("nod_2", formed=False)

    def it_is_not_one_that_asks_after_the_door_was_given_to_another() -> None:
        runtime = _runtime()
        runtime.claim("nod_2")
        runtime.claim("nod_0")

        assert runtime.opens("nod_2", formed=False)
        assert not runtime.opens("nod_0", formed=False)

    def it_is_nobody_once_a_worker_is_up_to_knock_on() -> None:
        runtime = _runtime()
        runtime.track("nod_2", _machinery("10.0.0.3", tunnel=40000))

        assert not runtime.opens("nod_0", formed=False)

    def it_is_nobody_on_a_compute_that_was_already_running_before_this_daemon() -> None:
        """A daemon that restarted holds nothing, and every machine looks like the first.

        The cluster is out there with fourteen workers in it, and the one node whose
        worker died is about to be started again — pointed at them, not told to open
        a second cluster of its own.
        """
        runtime = _runtime()
        runtime.claim("nod_2")

        assert not runtime.opens("nod_2", formed=True)

    def it_is_another_machine_when_the_one_that_was_given_the_door_never_opened_it() -> None:
        """Its connect died on the way, and a door nobody is standing in is not a door."""
        runtime = _runtime()
        runtime.claim("nod_2")
        assert runtime.opens("nod_2", formed=False)
        runtime.release("nod_2")

        runtime.claim("nod_0")

        assert runtime.opens("nod_0", formed=False)

    def it_is_every_machine_of_a_compute_that_has_no_cluster() -> None:
        """One worker per machine, each its own; nobody knocks on anybody."""
        runtime = _runtime(cluster=False)
        runtime.claim("nod_2")

        assert runtime.opens("nod_0", formed=False)


def describe_a_compute_whose_machines_all_connect_at_once() -> None:
    async def exactly_one_worker_is_started_to_open_the_cluster(tmp_path: Path) -> None:
        """Fifteen machines, ready together, taken hold of together.

        The shape a real compute arrives in, and the one neither the unit of the
        decision nor a two-container pool reproduces: every connect claims its machine
        before any of them gets as far as asking who opens the cluster. None opening it
        is fifteen workers waiting out the seed timeout; two is a compute split in
        half.
        """
        connector, nodes, started = await _connecting(tmp_path, machines=15)

        async with asyncio.TaskGroup() as connects:
            for node in nodes:
                connects.create_task(connector.connect(node.compute_id, node.id))

        assert len(started) == 15, "every machine was taken hold of"
        assert sum(not seeds for seeds in started.values()) == 1, "and exactly one of them was told to open the cluster"

    async def the_others_are_pointed_at_the_machine_that_opens_it(tmp_path: Path) -> None:
        connector, nodes, started = await _connecting(tmp_path, machines=15)

        async with asyncio.TaskGroup() as connects:
            for node in nodes:
                connects.create_task(connector.connect(node.compute_id, node.id))

        addresses = {node.id: node.address for node in nodes}
        door = next(f"{addresses[node_id]}:{worker.PORT}" for node_id, seeds in started.items() if not seeds)
        assert all(door in seeds for seeds in started.values() if seeds), "the door is in everybody else's seed list"


def describe_whom_a_worker_knocks_on() -> None:
    def it_knocks_on_nobody_when_it_is_the_one_opening_the_cluster() -> None:
        rows = (_row("nod_0", 0, "10.0.0.1"), _row("nod_2", 2, "10.0.0.3"))

        assert _seeds(rows, rows[1], opens=True) == ()

    def it_knocks_on_the_machine_that_opened_the_cluster_although_it_ranks_higher() -> None:
        """The split this is here to prevent.

        Rank 2 got its address first and opened the cluster. Rank 0 arrives two
        seconds later and is now the lowest-ranked node with an address — which used
        to be the whole rule, and made it open a second cluster of its own.
        """
        rows = (_row("nod_0", 0, "10.0.0.1"), _row("nod_2", 2, "10.0.0.3"))

        assert _seeds(rows, rows[0], opens=False) == (f"10.0.0.3:{worker.PORT}",)

    def it_knocks_on_every_live_machine_that_has_an_address_but_itself() -> None:
        rows = (_row("nod_0", 0, "10.0.0.1"), _row("nod_1", 1, "10.0.0.2"), _row("nod_2", 2, "10.0.0.3"))

        assert _seeds(rows, rows[1], opens=False) == (f"10.0.0.1:{worker.PORT}", f"10.0.0.3:{worker.PORT}")

    def it_leaves_out_a_machine_that_is_not_alive_and_one_with_no_address_yet() -> None:
        rows = (
            _row("nod_0", 0, "10.0.0.1", state="failed"),
            _row("nod_1", 1, None),
            _row("nod_2", 2, "10.0.0.3"),
            _row("nod_3", 3, "10.0.0.4"),
        )

        assert _seeds(rows, rows[3], opens=False) == (f"10.0.0.3:{worker.PORT}",)

    def it_knocks_on_nobody_when_there_is_nobody_left_to_knock_on() -> None:
        rows = (_row("nod_0", 0, "10.0.0.1", state="failed"), _row("nod_2", 2, "10.0.0.3"))

        assert _seeds(rows, rows[1], opens=False) == (), "somebody has to open it, and everyone else is gone"


def describe_the_world_a_worker_is_told_about() -> None:
    def it_is_every_live_machine_that_has_an_address_in_rank_order() -> None:
        rows = (_row("nod_2", 2, "10.0.0.3"), _row("nod_0", 0, "10.0.0.1"), _row("nod_1", 1, None))

        assert _peers(rows) == ("10.0.0.1", "10.0.0.3")


def describe_waiting_for_a_seed_to_answer() -> None:
    async def it_settles_for_the_first_one_that_answers() -> None:
        """Casty joins through the first seed that answers, so one of them is enough.

        The others are machines that may still be installing their dependencies, and
        a worker that waited for all of them would wait for the slowest bootstrap in
        the compute before joining anything.
        """
        async with _listening() as answering:
            assert await worker.reachable([_SILENT, answering]) == (answering, _SILENT)

    async def it_waits_for_nobody_when_it_is_the_one_opening_the_cluster() -> None:
        async with asyncio.timeout(5):
            assert await worker.reachable([]) == ()


def describe_a_client_that_dialled_the_wrong_cluster() -> None:
    async def it_dials_again_when_a_worker_that_is_up_is_not_in_its_view(monkeypatch: pytest.MonkeyPatch) -> None:
        """What a split leaves behind: the daemon inside the cluster of one.

        Without another dial the miss is permanent — the client is built once and
        kept — and every task placed on the other fourteen machines dies waiting for
        a member that this client is never going to see.
        """
        runtime, dialled = await _dialling(monkeypatch, ("10.0.0.1",), ("10.0.0.1", "10.0.0.2"))

        found = await runtime.member("nod_1")

        assert found.addr == f"10.0.0.2:{worker.PORT}"
        assert dialled[0].closed, "the client that saw the wrong cluster is dropped, not kept beside the new one"

    async def it_dials_again_only_once_for_a_worker_that_is_in_no_cluster_at_all(monkeypatch: pytest.MonkeyPatch) -> None:
        """A dropped client takes every call riding it with it, so a dead worker gets one and no more.

        The second attempt placed on that node finds the same absence, and dialling
        again over it would cost the compute every function running on every other
        machine, over and over, to learn the same thing.
        """
        runtime, dialled = await _dialling(monkeypatch, ("10.0.0.1",), ("10.0.0.1",))

        for _ in range(2):
            with pytest.raises(ComputeNotConnectedError):
                await runtime.member("nod_1")

        assert len(dialled) == 2, "one dial to find out, and none after that"


_SILENT = "127.0.0.1:1"
"""A port nothing listens on, on a host that answers at once that nothing does."""


class _Recording(Runtimes):
    """The runtimes with the machine taken out: what each node would be started with, and no SSH."""

    def __init__(self, started: dict[str, tuple[str, ...]]) -> None:
        super().__init__(lambda *_: None, _quiet, _quiet, _quiet)
        self._started = started

    async def start(self, runtime: Runtime, node_id: str, machine: Machine, **built: object) -> None:
        self._started[node_id] = built["seeds"]  # type: ignore[assignment]
        runtime.track(node_id, _machinery(machine.host or ""))


class _View:
    """A casty client that sees whichever cluster it was dialled into."""

    def __init__(self, members: Sequence[str]) -> None:
        self._members = tuple(casty.Member(node_id=uuid.uuid4(), addr=f"{addr}:{worker.PORT}") for addr in members)
        self.closed = False

    def members(self) -> tuple[casty.Member, ...]:
        return self._members

    async def close(self) -> None:
        self.closed = True


def _runtime(cluster: bool = True) -> Runtime:
    return Runtime("cmp_test", "auto", "key", cluster)


async def _quiet(*_: object) -> None:
    pass


def _machinery(address: str, tunnel: int | None = None) -> Machinery:
    node = Machinery(
        Machine(id=f"m-{address}", state="running", host=address, private_host=address),
        compute="cmp_test",
        private_key="key",
        image=Image(),
        source=Source(arguments=("skyward",)),
        listener=lambda *_: None,
        output=_quiet,
        sample=_quiet,
        phase=_quiet,
    )
    node.tunnel = tunnel
    return node


def _row(node_id: str, rank: int, address: str | None, state: NodeState = "ready") -> Node:
    return Node(
        id=node_id,
        compute_id="cmp_test",
        generation=1,
        rank=rank,
        revision=1,
        desired="present",
        state=state,
        provider_binding={},
        created_at=datetime.now(UTC),
        address=address,
    )


async def _connecting(tmp_path: Path, machines: int) -> tuple[Connector, tuple[Node, ...], dict[str, tuple[str, ...]]]:
    """A connector over a compute whose machines all have an address, and nothing up yet.

    The dict it comes back with is the one thing any of this is about: the seeds each
    worker would have been started with, by node id.
    """
    await database(tmp_path / "skyward.sqlite")
    blobs = BlobStore()
    image = msgspec.structs.replace(SPEC.image, includes_sha256=await blobs.store(b"the code the caller sent from its own machine"))
    computes = ComputeStore(EventStore(), NodeStore())
    compute, _ = await computes.create(ComputeCreate(spec=msgspec.structs.replace(SPEC, image=image)), idempotency_key="cluster")
    await computes.bind(compute.id, Infrastructure(offer=OFFER, private_key="key", binding={"skyward_cluster": True}))

    nodes = NodeStore()
    for rank in range(machines):
        requested = await nodes.request(compute.id, generation=1)
        address = f"10.0.0.{rank + 1}"
        await nodes.reachable(requested.id, Machine(id=f"m-{rank}", state="running", host=address, private_host=address))

    started: dict[str, tuple[str, ...]] = {}
    return Connector(computes, nodes, _Recording(started), blobs), await nodes.of(compute.id), started


async def _dialling(monkeypatch: pytest.MonkeyPatch, *clusters: Sequence[str]) -> tuple[Runtime, list[_View]]:
    """A runtime holding two ready nodes, seeing a different cluster on each dial.

    The list that comes back holds the clients it actually built, in the order it
    built them — the last cluster answers every dial after the ones listed.
    """
    views = iter([_View(members) for members in clusters])
    dialled: list[_View] = []

    async def connect(*_: object, **__: object) -> _View:
        match next(views, None):
            case None:
                dialled.append(dialled[-1])
            case view:
                dialled.append(view)
        return dialled[-1]

    monkeypatch.setattr(casty, "connect", connect)
    monkeypatch.setattr(runtimes_module, "MEMBERSHIP", 0.05)

    runtime = _runtime()
    for index, address in enumerate(("10.0.0.1", "10.0.0.2")):
        runtime.track(f"nod_{index}", _machinery(address, tunnel=40000 + index))
    return runtime, dialled


@asynccontextmanager
async def _listening(host: str = "127.0.0.1") -> AsyncIterator[str]:
    """A socket that accepts and hangs up, as a seed that is up looks from outside."""
    server = await asyncio.start_server(lambda _, writer: writer.close(), host, 0)
    async with server:
        yield f"{host}:{server.sockets[0].getsockname()[1]}"
