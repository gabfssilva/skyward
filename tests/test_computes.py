"""What the daemon refuses when a compute is asked for, and how it says so.

Both of these were a `500` before: an integrity error and a `RuntimeError` on
their way out through the exception handler. A caller can act on a refusal that
names itself — pick another name, ask the daemon that is holding the compute —
and can do nothing at all with an internal error.
"""

import asyncio
import sqlite3
import uuid
from collections import deque
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import msgspec
import pytest
from litestar.testing import AsyncTestClient

from skyward.server.application.machines import Machines
from skyward.server.application.mock import OFFER, SPEC
from skyward.server.application.node import Node as ApplicationNode
from skyward.server.application.runtimes import Files, Runtime, Runtimes
from skyward.server.application.source import Source
from skyward.server.http.app import create_app, services
from skyward.server.persistence.computes import ComputeStore, Infrastructure
from skyward.server.persistence.db import connect
from skyward.server.persistence.events import EventStore
from skyward.server.persistence.functions import BlobStore
from skyward.server.persistence.nodes import NodeStore
from skyward.server.persistence.offers import OfferCache
from skyward.server.persistence.providers import ProviderStore
from skyward.server.persistence.store import now
from skyward.server.persistence.tables import ComputeRow, EventRow, FunctionRow, NodeRow, TaskRow
from skyward.server.persistence.tasks import TaskStore
from skyward.shared.errors import ComputeNotConnectedError, NameTakenError, NotFoundError
from skyward.shared.events import ComputeAbandoned, ComputeDeleted
from skyward.shared.provider import Machine
from skyward.shared.schemas import Compute, ComputeCreate, DeletionCause, Image, Node, Task, TaskCreate, TaskOrder

pytestmark = pytest.mark.local


def describe_naming_a_compute() -> None:
    async def it_is_refused_when_another_compute_already_has_the_name(tmp_path: Path) -> None:
        await connect(tmp_path / "skyward.sqlite")
        store = ComputeStore(EventStore(), NodeStore())
        first, _ = await store.create(ComputeCreate(spec=SPEC, name="training"), idempotency_key="first")

        with pytest.raises(NameTakenError) as refused:
            await store.create(ComputeCreate(spec=SPEC, name="training"), idempotency_key="second")

        assert refused.value.details["compute"] == first.id
        assert refused.value.status == 409

    async def it_is_free_again_once_that_compute_is_deleted(tmp_path: Path) -> None:
        store = await _store(tmp_path)
        first, _ = await store.create(ComputeCreate(spec=SPEC, name="training"), idempotency_key="first")
        await _delete(store, first.id)

        second, _ = await store.create(ComputeCreate(spec=SPEC, name="training"), idempotency_key="second")

        assert second.id != first.id
        assert (await store.get("training")).id == second.id, "the name resolves to the compute that is alive"
        assert (await store.get(first.id)).status.state == "deleted", "the old one is still there by id"

    async def it_is_still_taken_while_that_compute_is_deleting(tmp_path: Path) -> None:
        store = await _store(tmp_path)
        first, _ = await store.create(ComputeCreate(spec=SPEC, name="training"), idempotency_key="first")
        await store.delete(first.id, first.revision, "delete")

        with pytest.raises(NameTakenError):
            await store.create(ComputeCreate(spec=SPEC, name="training"), idempotency_key="second")

    async def a_file_that_held_the_name_unique_forever_is_relaxed(tmp_path: Path) -> None:
        path = tmp_path / "skyward.sqlite"
        with sqlite3.connect(path) as old:
            old.execute(OLD_COMPUTES)
            old.execute("CREATE INDEX computes_name ON computes (name)")
            old.execute("INSERT INTO computes (id, name, status_state) VALUES ('cmp_old', 'training', 'deleted')")

        await connect(path)
        store = ComputeStore(EventStore(), NodeStore())
        fresh, _ = await store.create(ComputeCreate(spec=SPEC, name="training"), idempotency_key="again")

        assert (await store.get("training")).id == fresh.id
        assert await ComputeRow.select(ComputeRow.id).where(ComputeRow.id == "cmp_old").first(), "the rows survived the rebuild"
        with pytest.raises(sqlite3.IntegrityError):
            await ComputeRow(id="cmp_dup", name="training", status_state="requested").save().run()

    async def it_lets_two_computes_go_unnamed(tmp_path: Path) -> None:
        await connect(tmp_path / "skyward.sqlite")
        store = ComputeStore(EventStore(), NodeStore())

        first, _ = await store.create(ComputeCreate(spec=SPEC), idempotency_key="first")
        second, _ = await store.create(ComputeCreate(spec=SPEC), idempotency_key="second")

        assert first.id != second.id, "a name nobody gave is not a name two computes share"


def describe_listing_computes() -> None:
    async def an_empty_page_has_no_cursor(tmp_path: Path) -> None:
        store = await _store(tmp_path)

        page = await store.list(cursor=None, limit=0, state=None, owned=None, live=None)

        assert page.items == () and page.next_cursor is None

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
        await _delete(store, gone.id)

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

    async def they_are_asked_for_by_why_they_ended(tmp_path: Path) -> None:
        store = await _store(tmp_path)
        asked, _ = await store.create(ComputeCreate(spec=SPEC, name="asked"), idempotency_key="k1")
        left, _ = await store.create(ComputeCreate(spec=SPEC, name="left"), idempotency_key="k2")
        await _delete(store, asked.id)
        await _delete(store, left.id, "abandoned")

        page = await store.list(None, 50, None, None, None, "abandoned")

        assert [compute.name for compute in page.items] == ["left"]

    async def a_page_says_how_many_there_are_to_walk(tmp_path: Path) -> None:
        store = await _store(tmp_path)
        for index in range(3):
            await store.create(ComputeCreate(spec=SPEC, name=f"c{index}"), idempotency_key=f"k{index}")

        page = await store.list(None, 2, None, None, None)

        assert len(page.items) == 2 and page.total == 3, "what the filters match, not what the page carries"


def describe_what_a_compute_has_cost() -> None:
    async def it_is_its_machines_bill_up_to_now_while_it_is_live(tmp_path: Path) -> None:
        store = await _store(tmp_path)
        compute, _ = await store.create(ComputeCreate(spec=SPEC), idempotency_key="running")
        nodes = NodeStore()
        node = await nodes.request(compute.id, compute.generation)
        await NodeRow.update({
            NodeRow.launched_at: now() - timedelta(minutes=90),
            NodeRow.price_per_hour: 2.0,
            NodeRow.billing_unit: "hour",
        }).where(NodeRow.id == node.id).run()
        await nodes.request(compute.id, compute.generation)

        assert (await store.get(compute.id)).cost == pytest.approx(2 * 2.0)

    async def a_deleted_one_charges_no_further_and_agrees_with_its_ending(tmp_path: Path) -> None:
        store = await _store(tmp_path)
        compute, _ = await store.create(ComputeCreate(spec=SPEC), idempotency_key="closed")
        nodes = NodeStore()
        node = await nodes.request(compute.id, compute.generation)
        launched = now() - timedelta(hours=4)
        await NodeRow.update({
            NodeRow.launched_at: launched,
            NodeRow.terminated_at: launched + timedelta(minutes=90),
            NodeRow.price_per_hour: 2.0,
            NodeRow.billing_unit: "hour",
        }).where(NodeRow.id == node.id).run()

        await _delete(store, compute.id)

        served = await store.get(compute.id)
        assert served.ended is not None
        assert served.cost == pytest.approx(2 * 2.0) == served.ended.cost


def describe_a_compute_that_has_ended() -> None:
    async def a_live_one_has_no_ending(tmp_path: Path) -> None:
        store = await _store(tmp_path)
        compute, _ = await store.create(ComputeCreate(spec=SPEC), idempotency_key="live")

        assert (await store.get(compute.id)).ended is None

    async def it_says_when_its_last_machine_was_gone_and_that_somebody_asked(tmp_path: Path) -> None:
        store = await _store(tmp_path)
        compute, _ = await store.create(ComputeCreate(spec=SPEC), idempotency_key="asked")
        before = now()

        await _delete(store, compute.id)

        ended = (await store.get(compute.id)).ended
        assert ended is not None
        assert ended.cause == "requested" and ended.at >= before

    async def it_says_it_was_abandoned_when_the_reconciler_let_it_go(tmp_path: Path) -> None:
        store = await _store(tmp_path)
        compute, _ = await store.create(ComputeCreate(spec=SPEC), idempotency_key="left")

        await _delete(store, compute.id, "abandoned")

        ended = (await store.get(compute.id)).ended
        assert ended is not None and ended.cause == "abandoned"

    async def the_first_cause_given_is_the_one_kept(tmp_path: Path) -> None:
        store = await _store(tmp_path)
        compute, _ = await store.create(ComputeCreate(spec=SPEC), idempotency_key="twice")
        await store.delete(compute.id, compute.revision, "reclaimed", "abandoned")

        await _delete(store, compute.id)

        ended = (await store.get(compute.id)).ended
        assert ended is not None and ended.cause == "abandoned"

    async def its_bill_is_its_machines_and_its_calls_are_counted_by_how_they_ended(tmp_path: Path) -> None:
        store = await _store(tmp_path)
        compute, _ = await store.create(ComputeCreate(spec=SPEC), idempotency_key="billed")
        tasks, nodes = TaskStore(store, NodeStore(), BlobStore()), NodeStore()
        for outcome in ("succeeded", "failed", "timed_out", "succeeded"):
            task = await _submit(tasks, compute.id)
            await TaskRow.update({TaskRow.state: outcome}).where(TaskRow.id == task.id).run()
        launched = now() - timedelta(hours=3)
        for held, price, unit in ((timedelta(minutes=61), 2.0, "hour"), (timedelta(seconds=90), 3.6, "second")):
            node = await nodes.request(compute.id, compute.generation)
            await NodeRow.update({
                NodeRow.launched_at: launched,
                NodeRow.terminated_at: launched + held,
                NodeRow.price_per_hour: price,
                NodeRow.billing_unit: unit,
            }).where(NodeRow.id == node.id).run()
        await nodes.request(compute.id, compute.generation)

        await _delete(store, compute.id)

        ended = (await store.get(compute.id)).ended
        assert ended is not None
        assert ended.cost == pytest.approx(2 * 2.0 + 90 / 3600 * 3.6)
        assert (ended.calls, ended.failed) == (4, 2)

    async def a_page_carries_the_ending_of_each_deleted_compute_on_it(tmp_path: Path) -> None:
        store = await _store(tmp_path)
        gone, _ = await store.create(ComputeCreate(spec=SPEC, name="gone"), idempotency_key="k1")
        await store.create(ComputeCreate(spec=SPEC, name="running"), idempotency_key="k2")
        await _delete(store, gone.id)

        page = await store.list(None, 50, None, None, None)

        assert {compute.name: compute.ended is not None for compute in page.items} == {"gone": True, "running": False}

    async def one_deleted_before_its_row_kept_how_is_told_from_the_log(tmp_path: Path) -> None:
        store = await _store(tmp_path)
        compute, _ = await store.create(ComputeCreate(spec=SPEC), idempotency_key="aged")
        await store.apply(ComputeAbandoned(compute=compute.id))
        await _delete(store, compute.id)
        said = await EventRow.select(EventRow.created_at).where((EventRow.compute_id == compute.id) & (EventRow.type == "compute.deleted")).first()
        await ComputeRow.update({ComputeRow.deleted_at: None, ComputeRow.deletion_cause: None}).where(ComputeRow.id == compute.id).run()

        await connect(tmp_path / "skyward.sqlite")

        ended = (await ComputeStore(EventStore(), NodeStore()).get(compute.id)).ended
        assert said is not None and ended is not None
        assert (ended.at, ended.cause) == (said["created_at"], "abandoned")


def describe_listing_a_computes_tasks() -> None:
    async def they_come_newest_first_each_with_its_own_attempts(tmp_path: Path) -> None:
        tasks, compute = await _tasks(tmp_path)
        submitted = [await _submit(tasks, compute) for _ in range(3)]
        await tasks.observe(submitted[1].executions[0].id, "failed", again=True)

        page = await tasks.list(None, 2, compute)

        assert page.items == (await tasks.get(submitted[2].id), await tasks.get(submitted[1].id)), "the latest two, not the first two"
        assert [len(task.executions) for task in page.items] == [1, 2]

    async def a_page_picks_up_below_the_one_before_it(tmp_path: Path) -> None:
        tasks, compute = await _tasks(tmp_path)
        submitted = [await _submit(tasks, compute) for _ in range(4)]

        first = await tasks.list(None, 2, compute)
        second = await tasks.list(first.next_cursor, 2, compute)

        assert [task.id for task in first.items] == [submitted[3].id, submitted[2].id]
        assert [task.id for task in second.items] == [submitted[1].id, submitted[0].id]

    async def a_page_says_how_many_tasks_match(tmp_path: Path) -> None:
        tasks, compute = await _tasks(tmp_path)
        for _ in range(3):
            await _submit(tasks, compute)

        page = await tasks.list(None, 2, compute)

        assert len(page.items) == 2 and page.total == 3

    async def it_keeps_the_tasks_in_any_of_the_states_asked_for(tmp_path: Path) -> None:
        tasks, compute = await _tasks(tmp_path)
        queued, running, failed = [await _submit(tasks, compute) for _ in range(3)]
        await tasks.observe(running.executions[0].id, "started")
        await tasks.observe(failed.executions[0].id, "failed")

        page = await tasks.list(None, 10, compute, states=("running", "failed"))

        assert {task.id for task in page.items} == {running.id, failed.id} and page.total == 2
        assert queued.id not in {task.id for task in page.items}

    async def a_function_is_asked_for_by_name_whatever_code_was_uploaded_under_it(tmp_path: Path) -> None:
        tasks, compute = await _tasks(tmp_path)
        for sha256, name in (("a" * 64, "fill"), ("b" * 64, "fill"), ("c" * 64, "ping")):
            await FunctionRow(sha256=sha256, size_bytes=1, codec="cloudpickle", name=name, created_at=now()).save().run()
        first, second, _ = [await _submit(tasks, compute, function) for function in ("a" * 64, "b" * 64, "c" * 64)]

        page = await tasks.list(None, 10, compute, function="fill")

        assert {task.id for task in page.items} == {first.id, second.id} and page.total == 2

    async def by_state_it_runs_running_then_queued_next_to_run_first_then_the_latest_to_finish(tmp_path: Path) -> None:
        tasks, compute = await _tasks(tmp_path)
        board = await _board(tasks, compute)

        page = await tasks.list(None, 10, compute, order="state")

        expected = ("running late", "running early", "queued early", "queued late", "finished late", "finished early")
        assert [task.id for task in page.items] == [board[name] for name in expected]

    async def by_finished_it_puts_the_latest_to_finish_first_and_the_unfinished_last(tmp_path: Path) -> None:
        tasks, compute = await _tasks(tmp_path)
        board = await _board(tasks, compute)

        page = await tasks.list(None, 10, compute, order="finished")

        expected = ("finished late", "finished early", "queued late", "running late", "queued early", "running early")
        assert [task.id for task in page.items] == [board[name] for name in expected]

    @pytest.mark.parametrize("order", ["submitted", "state", "finished"])
    async def a_walk_a_page_at_a_time_meets_every_task_once_in_order(tmp_path: Path, order: TaskOrder) -> None:
        tasks, compute = await _tasks(tmp_path)
        await _board(tasks, compute)
        whole = [task.id for task in (await tasks.list(None, 50, compute, order=order)).items]

        walked: list[str] = []
        page = await tasks.list(None, 2, compute, order=order)
        walked += [task.id for task in page.items]
        await _submit(tasks, compute)
        while page.next_cursor:
            page = await tasks.list(page.next_cursor, 2, compute, order=order)
            walked += [task.id for task in page.items]

        assert len(walked) == len(set(walked)), "no task twice"
        assert [task for task in walked if task in whole] == whole, "a task submitted mid-walk does not shift where a held cursor picks up"

    async def a_cursor_is_only_good_for_the_order_it_came_from(tmp_path: Path) -> None:
        tasks, compute = await _tasks(tmp_path)
        await _board(tasks, compute)
        page = await tasks.list(None, 2, compute, order="state")

        with pytest.raises(NotFoundError):
            await tasks.list(page.next_cursor, 2, compute, order="finished")

    async def the_endpoint_takes_state_more_than_once_and_an_order(tmp_path: Path) -> None:
        _, compute = await _tasks(tmp_path)
        svc = services()
        async with AsyncTestClient(app=create_app(svc, logging=False)) as http:
            assert isinstance(svc.tasks, TaskStore)
            board = await _board(svc.tasks, compute)

            answer = await http.get("/v1/tasks", params=[("compute", compute), ("state", "queued"), ("state", "running"), ("order", "state")])

        assert answer.status_code == 200, answer.text
        assert [task["id"] for task in answer.json()["items"]] == [board[name] for name in ("running late", "running early", "queued early", "queued late")]
        assert answer.json()["total"] == 4


def describe_reaching_a_compute_this_daemon_is_not_holding() -> None:
    async def it_is_told_rather_than_raised_through(tmp_path: Path) -> None:
        async def quiet(*_: object) -> None:
            pass

        files = Files(Runtimes(listener=lambda *_: None, output=quiet, sample=quiet, phase=quiet))

        with pytest.raises(ComputeNotConnectedError) as refused:
            await files.run("cmp_elsewhere", "all", "echo hello")

        assert refused.value.status == 409
        assert refused.value.retryable, "another daemon holds it, or this one has not picked it up yet"


def describe_a_machine_that_is_bought_and_never_says_where_it_is() -> None:
    async def it_is_given_up_on_once_the_window_closes(tmp_path: Path) -> None:
        machines, compute, node = await _bought(tmp_path / "skyward.sqlite", provision_timeout=0.01)
        await asyncio.sleep(0.05)

        await machines.resolve(compute)

        given_up = await NodeStore().get(compute.id, node.id)
        assert given_up.state == "lost"
        assert given_up.last_error is not None
        assert "never published an address" in given_up.last_error.message

    async def it_is_waited_on_while_the_window_is_open(tmp_path: Path) -> None:
        machines, compute, node = await _bought(tmp_path / "skyward.sqlite", provision_timeout=600.0)

        await machines.resolve(compute)

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
            await machines.resolve(compute)

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
            await machines.resolve(compute)

        given_up = await NodeStore().get(compute.id, node.id)
        assert given_up.state == "lost"
        assert given_up.last_error is not None
        assert "downloading (55%)" in given_up.last_error.message, "the reason names what the machine was doing"


def describe_binding_a_compute() -> None:
    async def it_is_served_with_the_offer_it_was_bound_to(tmp_path: Path) -> None:
        store = await _store(tmp_path)
        compute, _ = await store.create(ComputeCreate(spec=SPEC), idempotency_key="bound")

        await store.bind(compute.id, Infrastructure(offer=OFFER, offer_id=OFFER.id, provider_id="prv_1"))

        assert compute.offer is None
        assert (await store.get(compute.id)).offer == OFFER

    async def the_first_key_written_is_the_key_kept(tmp_path: Path) -> None:
        """Two daemons on one file race their minted pairs; the machines trust the winner's."""
        store = await _store(tmp_path)
        compute, _ = await store.create(ComputeCreate(spec=SPEC), idempotency_key="raced")

        await store.bind(compute.id, Infrastructure(offer=OFFER, offer_id=OFFER.id, provider_id="prv_1", binding={"a": 1}, private_key="winner"))
        await store.bind(compute.id, Infrastructure(offer=OFFER, offer_id=OFFER.id, provider_id="prv_1", binding={"b": 2}, private_key="loser"))

        stored = await store.infrastructure(compute.id)
        assert stored.private_key == "winner"
        assert stored.binding == {"a": 1}, "a binding the fleet was launched under is not overwritten"

    async def a_rebind_carrying_the_same_key_lands(tmp_path: Path) -> None:
        """Relocation binds the compute into another region under the key it already has."""
        store = await _store(tmp_path)
        compute, _ = await store.create(ComputeCreate(spec=SPEC), idempotency_key="moved")

        await store.bind(compute.id, Infrastructure(offer=OFFER, offer_id=OFFER.id, provider_id="prv_1", binding={"region": "us"}, private_key="key"))
        await store.bind(compute.id, Infrastructure(offer=OFFER, offer_id=OFFER.id, provider_id="prv_1", binding={"region": "eu"}, private_key="key"))

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
        store = ComputeStore(EventStore(), NodeStore())
        compute, _ = await store.create(ComputeCreate(spec=SPEC), idempotency_key="aged")
        await ComputeRow.raw(
            "UPDATE computes SET spec = json_remove(json_set(spec, '$.nodes.desired', "
            "json_extract(spec, '$.nodes.initial')), '$.nodes.initial')"
        ).run()

        await connect(database)

        mended = await ComputeStore(EventStore(), NodeStore()).get(compute.id)
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
    nodes, providers = NodeStore(), ProviderStore()
    computes = ComputeStore(EventStore(), nodes)
    spec = msgspec.structs.replace(SPEC, options=msgspec.structs.replace(SPEC.options, provision_timeout=provision_timeout))
    compute, _ = await computes.create(ComputeCreate(spec=spec), idempotency_key="bought")
    await computes.bind(compute.id, Infrastructure(offer=OFFER, offer_id=OFFER.id, provider_id="prv_1", binding={"prefix": "skyward-"}))

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


OLD_COMPUTES = """CREATE TABLE "computes" ("id" VARCHAR(255) PRIMARY KEY NOT NULL DEFAULT '', "name" VARCHAR(255) UNIQUE DEFAULT null,
"revision" INTEGER NOT NULL DEFAULT 1, "generation" INTEGER NOT NULL DEFAULT 1, "spec" JSONB NOT NULL DEFAULT '{}',
"provider_id" VARCHAR(255) DEFAULT null, "offer_id" VARCHAR(255) DEFAULT null, "offer" JSONB DEFAULT null,
"binding" JSONB NOT NULL DEFAULT '{}', "private_key" TEXT DEFAULT null, "markets" JSONB NOT NULL DEFAULT '[]',
"volumes" JSONB NOT NULL DEFAULT '[]', "status_state" VARCHAR(255) NOT NULL DEFAULT '',
"status_observed_generation" INTEGER NOT NULL DEFAULT 0, "status_nodes_ready" INTEGER NOT NULL DEFAULT 0,
"status_nodes_total" INTEGER NOT NULL DEFAULT 0, "status_drift" JSONB NOT NULL DEFAULT '[]', "status_error" JSONB DEFAULT null,
"lease_owner" VARCHAR(255) DEFAULT null, "lease_expires_at" TIMESTAMPTZ DEFAULT null,
"created_at" TIMESTAMPTZ NOT NULL DEFAULT current_timestamp, "authority" JSONB DEFAULT null)"""
"""The ``computes`` table as a daemon wrote it while a name was unique forever."""


async def _store(tmp_path: Path) -> ComputeStore:
    await connect(tmp_path / "skyward.sqlite")
    return ComputeStore(EventStore(), NodeStore())


async def _delete(store: ComputeStore, compute: str, cause: DeletionCause = "requested") -> None:
    """Ask for it to go, and then say that it went — what the reconciler says once its machines are gone."""
    await store.delete(compute, (await store.get(compute)).revision, f"delete:{compute}", cause)
    await store.apply(ComputeDeleted(compute=compute))


async def _tasks(tmp_path: Path) -> tuple[TaskStore, str]:
    """A task store, and a compute of its own to submit to."""
    store = await _store(tmp_path)
    compute, _ = await store.create(ComputeCreate(spec=SPEC), idempotency_key="compute")
    return TaskStore(store, NodeStore(), BlobStore()), compute.id


async def _submit(tasks: TaskStore, compute: str, function: str = "f" * 64) -> Task:
    task, _ = await tasks.submit(TaskCreate(compute=compute, function=function, dispatch="one", args_inline=b"args"), idempotency_key=uuid.uuid4().hex)
    return task


async def _board(tasks: TaskStore, compute: str) -> dict[str, str]:
    """Two tasks running, two queued and two finished, submitted a minute apart in shuffled order, each finished one a minute apart."""
    start = now() - timedelta(hours=1)
    board: dict[str, str] = {}
    for name in ("queued late", "finished early", "running early", "queued early", "running late", "finished late"):
        board[name] = (await _submit(tasks, compute)).id
    for minute, name in enumerate(("finished early", "running early", "queued early", "finished late", "running late", "queued late")):
        await TaskRow.update({TaskRow.submitted_at: start + timedelta(minutes=minute)}).where(TaskRow.id == board[name]).run()
    for name in ("running early", "running late", "finished early", "finished late"):
        (execution,) = (await tasks.get(board[name])).executions
        await tasks.observe(execution.id, "started")
    for minute, name in enumerate(("finished early", "finished late"), start=10):
        (execution,) = (await tasks.get(board[name])).executions
        await tasks.observe(execution.id, "succeeded")
        await TaskRow.update({TaskRow.finished_at: start + timedelta(minutes=minute)}).where(TaskRow.id == board[name]).run()
    return board


def _runtime(cluster: bool = True) -> Runtime:
    return Runtime("cmp_1", "pypi", private_key="key", cluster=cluster)


def _node(host: str = "127.0.0.1") -> ApplicationNode:
    async def quiet(*_: object) -> None:
        pass

    return ApplicationNode(
        Machine(id="m1", state="running", host=host),
        compute="cmp_1",
        private_key="key",
        image=Image(),
        source=Source(arguments=("skyward",)),
        listener=lambda *_: None,
        output=quiet,
        sample=quiet,
        phase=quiet,
    )
