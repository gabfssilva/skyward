"""One pass of the reconciler: what it reads, and what it lets go of."""

import asyncio
import uuid
from datetime import timedelta
from pathlib import Path

import pytest

from skyward.server.application.connector import Connector
from skyward.server.application.dispatcher import Dispatcher
from skyward.server.application.machines import Machines
from skyward.server.application.mock import SPEC
from skyward.server.application.reconciler import ABANDON_SECONDS, Reconciler, Wakeup
from skyward.server.application.runtimes import Runtimes
from skyward.server.http.emitter import ReconcilingEventEmitter
from skyward.server.http.listeners import build_listeners
from skyward.server.persistence.computes import ComputeStore, GenerationStore
from skyward.server.persistence.events import EventStore
from skyward.server.persistence.functions import BlobStore
from skyward.server.persistence.nodes import NodeStore
from skyward.server.persistence.offers import OfferCache
from skyward.server.persistence.providers import ProviderStore
from skyward.server.persistence.store import now
from skyward.server.persistence.tables import ComputeRow, EventRow
from skyward.server.persistence.tasks import TaskStore
from skyward.shared.errors import TaskFailedError
from skyward.shared.events import ComputeDeleted
from skyward.shared.schemas import ComputeCreate, Node, Task, TaskCreate
from tests.conftest import given

pytestmark = pytest.mark.local


class CountingNodes(NodeStore):
    """A node store that counts how often the whole list is asked for."""

    def __init__(self) -> None:
        super().__init__()
        self.listed = 0

    async def of(self, compute_id: str) -> tuple[Node, ...]:
        self.listed += 1
        return await super().of(compute_id)


def describe_one_pass() -> None:
    async def it_reads_the_nodes_once_when_the_pass_changes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Once after the provider is asked what became of them; a pass that buys and drains nothing has no reason to read them again."""
        computes, compute, nodes, reconciler = await _reconciler(tmp_path, monkeypatch)
        await reconciler.compute(compute)
        nodes.listed = 0

        await reconciler.compute(compute)

        assert nodes.listed == 1

    async def it_forgets_a_compute_once_it_is_deleted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        computes, compute, nodes, reconciler = await _reconciler(tmp_path, monkeypatch)
        await reconciler.compute(compute)
        assert compute in reconciler._locks

        await computes.delete(compute, (await computes.get(compute)).revision, "once")
        for node in await nodes.of(compute):
            await nodes.observe(node.id, "deleted")
        await reconciler.compute(compute)

        assert (await computes.get(compute)).status.state == "deleted"
        assert compute not in reconciler._locks
        assert not any(node.id in reconciler._idle for node in await nodes.of(compute))


def describe_a_deleted_compute() -> None:
    async def it_cancels_what_it_still_held_and_answers_the_caller_waiting_on_it(tmp_path: Path) -> None:
        daemon = await _daemon(tmp_path)
        task = await daemon.submit()
        daemon.runtimes.open(daemon.compute, "pypi", "a private key")
        waiting = asyncio.create_task(daemon.tasks.result(task.id, wait_seconds=30))
        await asyncio.sleep(0.05)
        assert not waiting.done(), "nothing is ever going to run it, and the caller is waiting all the same"

        await daemon.delete()

        with pytest.raises(TaskFailedError) as answered:
            async with asyncio.timeout(5):
                await waiting
        assert answered.value.details["state"] == "cancelled"
        assert f"compute {daemon.compute} was deleted" in answered.value.message
        await daemon.quiet()
        assert daemon.runtimes.of(daemon.compute) is None, "its connections go with it, not when the daemon shuts down"

    async def it_answers_for_what_a_compute_deleted_earlier_left_behind(tmp_path: Path) -> None:
        daemon = await _daemon(tmp_path)
        queued, running = await daemon.submit(), await daemon.submit()
        await daemon.tasks.observe(running.executions[0].id, "started", node_id="nod_gone")
        await daemon.delete_unannounced()

        await daemon.tick()
        await daemon.quiet()

        assert (await daemon.tasks.get(queued.id)).state == "cancelled", "it never left the daemon, so it never ran"
        assert (await daemon.tasks.get(running.id)).state == "indeterminate", "it was on a machine, and what it did there is not known"
        assert await daemon.said(running.id) == ["task.indeterminate"]
        assert await daemon.reconciler.unsettled() == ((), ()), "answered for once, and never offered again"

    async def the_tick_offers_its_tasks_to_nobody(tmp_path: Path) -> None:
        daemon = await _daemon(tmp_path)
        await daemon.submit()
        alive, _ = await daemon.computes.create(ComputeCreate(spec=SPEC), idempotency_key="alive")
        kept, _ = await daemon.tasks.submit(TaskCreate(compute=alive.id, function="f" * 64, dispatch="one", args_inline=b"args"), idempotency_key="kept")
        await daemon.delete_unannounced()

        computes, tasks = await daemon.reconciler.unsettled()

        assert tasks == (kept.id,), "a task of a deleted compute has nothing left to run it"
        assert computes == (alive.id, daemon.compute), "the compute is offered in its place, until what it owed is answered for"

    async def it_is_not_offered_for_a_task_without_an_attempt_to_answer_for(tmp_path: Path) -> None:
        daemon = await _daemon(tmp_path)
        broadcast, _ = await daemon.tasks.submit(
            TaskCreate(compute=daemon.compute, function="f" * 64, dispatch="all", args_inline=b"args"),
            idempotency_key="broadcast",
        )
        await daemon.delete_unannounced()

        assert broadcast.executions == (), "a broadcast admitted while no node was ready is given no attempt"
        assert await daemon.reconciler.unsettled() == ((), ()), "its deletion has nothing to answer, so there is nothing to offer it for on every tick"


def describe_a_daemon_going_away() -> None:
    async def it_answers_a_result_being_waited_on_with_no_outcome_yet(tmp_path: Path) -> None:
        daemon = await _daemon(tmp_path)
        task = await daemon.submit()
        waiting = asyncio.create_task(daemon.tasks.result(task.id, wait_seconds=30))
        await asyncio.sleep(0.05)
        assert not waiting.done()

        daemon.tasks.close()

        async with asyncio.timeout(1):
            assert await waiting is None, "not a verdict: the caller asks again, of whichever daemon is there by then"


def describe_a_compute_whose_lease_ran_out() -> None:
    async def while_the_daemon_was_down_its_owner_gets_the_minute_to_renew_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        computes, compute, _, reconciler = await _reconciler(tmp_path, monkeypatch)
        await _unrenewed(compute, seconds=300)

        await reconciler.compute(compute)

        assert (await computes.get(compute)).spec.desired != "deleted", "an owner cannot renew through a daemon that was not there"

    async def once_this_daemon_has_been_up_that_long_nobody_is_coming(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        computes, compute, _, reconciler = await _reconciler(tmp_path, monkeypatch)
        await _unrenewed(compute, seconds=300)
        monkeypatch.setattr(reconciler, "_up_since", now() - timedelta(seconds=ABANDON_SECONDS + 1))

        await reconciler.compute(compute)

        assert (await computes.get(compute)).spec.desired == "deleted"


async def _unrenewed(compute: str, seconds: float) -> None:
    """Make the compute older than the newborn grace, with a lease its owner last renewed ``seconds`` ago."""
    await ComputeRow.update(
        {
            ComputeRow.created_at: now() - timedelta(seconds=seconds + ABANDON_SECONDS),
            ComputeRow.lease_owner: "sdk_gone",
            ComputeRow.lease_expires_at: now() - timedelta(seconds=seconds),
        }
    ).where(ComputeRow.id == compute).run()


async def _reconciler(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ComputeStore, str, CountingNodes, Reconciler]:
    """A reconciler over a compute of its own, with a provider that has nothing to say."""
    events = EventStore()
    computes, compute = await given(tmp_path / "skyward.sqlite", events=events)
    nodes, blobs = CountingNodes(), BlobStore()
    machines = Machines(computes, nodes, ProviderStore(), OfferCache(ProviderStore()), blobs, events)

    async def nothing(*_: object) -> None:
        return None

    monkeypatch.setattr(machines, "resolve", nothing)
    reconciler = Reconciler(computes, GenerationStore(computes), nodes, TaskStore(computes, nodes, blobs), machines, events, Wakeup())
    return computes, compute.id, nodes, reconciler


class _Daemon:
    """The control plane over one compute, wired as the app wires it: the wakeups reach the listeners through the bus."""

    def __init__(
        self,
        computes: ComputeStore,
        tasks: TaskStore,
        runtimes: Runtimes,
        reconciler: Reconciler,
        bus: ReconcilingEventEmitter,
        compute: str,
    ) -> None:
        self.computes, self.tasks, self.runtimes, self.reconciler, self.bus, self.compute = computes, tasks, runtimes, reconciler, bus, compute

    async def submit(self) -> Task:
        task, _ = await self.tasks.submit(
            TaskCreate(compute=self.compute, function="f" * 64, dispatch="one", args_inline=b"args"),
            idempotency_key=uuid.uuid4().hex,
        )
        return task

    async def delete(self) -> None:
        """What deleting the compute sets going, as far as the pass that finds no machine left."""
        await self.computes.delete(self.compute, (await self.computes.get(self.compute)).revision, "delete")
        await self.reconciler.compute(self.compute)

    async def delete_unannounced(self) -> None:
        """The compute as a daemon that answered for nothing left it: deleted, and nobody told."""
        await self.computes.delete(self.compute, (await self.computes.get(self.compute)).revision, "delete")
        await self.computes.apply(ComputeDeleted(compute=self.compute))

    async def tick(self) -> None:
        """What the daemon's clock offers on every turn, without the clock."""
        computes, tasks = await self.reconciler.unsettled()
        for compute_id in computes:
            self.bus.emit("compute.changed", compute_id=compute_id)
        for task_id in tasks:
            self.bus.emit("task.changed", task_id=task_id)

    async def quiet(self) -> None:
        """Until every wakeup emitted so far has run, and every wakeup those emitted in turn."""
        async with asyncio.timeout(5):
            while self.bus._running or self.bus._loose:
                await asyncio.sleep(0.01)

    async def said(self, task: str) -> list[str]:
        rows = await EventRow.select(EventRow.type).where(EventRow.task_id == task).order_by(EventRow.sequence)
        return [row["type"] for row in rows]


async def _daemon(tmp_path: Path) -> _Daemon:
    events = EventStore()
    computes, compute = await given(tmp_path / "skyward.sqlite", events=events)
    nodes, blobs = NodeStore(), BlobStore()
    tasks = TaskStore(computes, nodes, blobs)
    machines = Machines(computes, nodes, ProviderStore(), OfferCache(ProviderStore()), blobs, events)

    async def quiet(*_: object) -> None:
        pass

    runtimes = Runtimes(listener=lambda *_: None, output=quiet, sample=quiet, phase=quiet)
    wake = Wakeup()
    reconciler = Reconciler(computes, GenerationStore(computes), nodes, tasks, machines, events, wake)
    dispatcher = Dispatcher(computes, tasks, nodes, blobs, events, runtimes, wake)
    bus = ReconcilingEventEmitter(build_listeners(reconciler, dispatcher, machines, Connector(computes, nodes, runtimes, blobs)))
    wake.bind(bus.emit)
    return _Daemon(computes, tasks, runtimes, reconciler, bus, compute.id)
