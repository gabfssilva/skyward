"""What ``/v1`` serves a compute, a node and a task as, and what ``include`` adds to them.

A client drawing a compute has what it needs in one answer: the node holding each
rank, the account it was bought from, what it costs right now and what each machine
is doing. What is expensive, or changes by the second, is asked for by name — and is
absent otherwise, which is how an answer says it was not asked rather than that
there was nothing.
"""

import hashlib
import os
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import pytest
from litestar.testing import AsyncTestClient
from msgspec import NODEFAULT, Struct, structs

from skyward.api import v1
from skyward.server.application.mock import OFFER, SPEC
from skyward.server.http.app import Services, create_app, services
from skyward.server.persistence.computes import ComputeStore, Infrastructure
from skyward.server.persistence.db import connect
from skyward.server.persistence.events import EventStore
from skyward.server.persistence.nodes import NodeStore
from skyward.server.persistence.store import now
from skyward.server.persistence.tasks import TaskStore
from skyward.shared import schemas
from skyward.shared.events import ConsoleEvent, PhaseEvent
from skyward.shared.provider import Machine
from skyward.shared.schemas import ComputeCreate, Error, MetricSample, PhaseMark, TaskCreate

pytestmark = pytest.mark.local

PASSWORD = "hunter2-root-password"


@dataclass(frozen=True, slots=True)
class Plane:
    """A compute with some history, and the ids a test asks about."""

    svc: Services
    compute: str
    held: str
    """The node holding rank 0, ready."""
    gone: str
    """The node that held rank 0 before it, given up on."""
    booting: str
    """Rank 1, bought and still without an address."""
    function: str
    running: str
    succeeded: str
    failed: str


@pytest.fixture
async def plane(tmp_path: Any) -> Plane:
    await connect(tmp_path / "skyward.sqlite")
    svc = services()
    computes, nodes, tasks, events = svc.computes, svc.nodes, svc.tasks, svc.events
    assert isinstance(computes, ComputeStore) and isinstance(nodes, NodeStore) and isinstance(tasks, TaskStore) and isinstance(events, EventStore)

    compute, _ = await computes.create(ComputeCreate(spec=SPEC, name="training"), idempotency_key="training")
    await computes.bind(compute.id, Infrastructure(offer=OFFER, offer_id=OFFER.id, provider_id=OFFER.provider_id))

    gone = await nodes.request(compute.id, compute.generation)
    await nodes.launched(gone.id, Machine(id="m0", state="running", host="198.51.100.1", password=PASSWORD), offer=OFFER, market="spot")
    await nodes.observe(gone.id, "lost", Error(code="not_found", message="the provider stopped listing it", retryable=True))
    held = await nodes.request(compute.id, compute.generation)
    machine = Machine(id="m1", state="running", host="203.0.113.7", port=2200, user="ubuntu", password=PASSWORD)
    await nodes.launched(held.id, machine, offer=OFFER, market="spot")
    await nodes.observe(held.id, "ready")
    booting = await nodes.request(compute.id, compute.generation)
    await nodes.launched(booting.id, Machine(id="m2", state="pending", progress="pulling image", completion=0.4), offer=OFFER, market="spot")

    code = os.urandom(1024)
    function = hashlib.sha256(code).hexdigest()
    await svc.functions.register(function, code, "train")

    async def submitted() -> str:
        task, _ = await tasks.submit(TaskCreate(compute=compute.id, function=function, dispatch="one", args_inline=b"args"), idempotency_key=uuid.uuid4().hex)
        (attempt,) = task.executions
        await tasks.observe(attempt.id, "started", node_id=held.id)
        return task.id

    running, succeeded, failed = await submitted(), await submitted(), await submitted()
    (won,) = (await tasks.get(succeeded)).executions
    await tasks.observe(won.id, "succeeded")
    (lost,) = (await tasks.get(failed)).executions
    await tasks.observe(lost.id, "failed", error=Error(code="task_failed", message="ZeroDivisionError", retryable=False))

    at = time.time_ns() // 1_000_000
    svc.metrics.add(compute.id, (MetricSample(node=held.id, name="gpu_util", at=at, value=50.0), MetricSample(node=held.id, name="cpu", at=at, value=20.0)))
    await svc.metrics.flush()

    marks: tuple[tuple[PhaseMark, str], ...] = (("started", "apt"), ("completed", "apt"), ("started", "uv"))
    for mark, phase in marks:
        await events.record(PhaseEvent(compute=compute.id, node=held.id, event=mark, phase=phase, at=now()))
    await events.record_all(tuple(ConsoleEvent(compute=compute.id, node=held.id, content=line) for line in ("one", "two")))

    return Plane(svc, compute.id, held.id, gone.id, booting.id, function, running, succeeded, failed)


@pytest.fixture
async def http(plane: Plane) -> AsyncIterator[AsyncTestClient]:
    async with AsyncTestClient(app=create_app(plane.svc, logging=False)) as client:
        yield client


async def _read(http: AsyncTestClient, path: str) -> Any:
    answer = await http.get(path)
    assert answer.status_code == 200, answer.text
    return answer.json()


def _by_id(nodes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {node["id"]: node for node in nodes}


def describe_reading_a_compute() -> None:
    async def it_carries_the_node_holding_each_rank(http: AsyncTestClient, plane: Plane) -> None:
        compute = await _read(http, "/v1/computes/training")

        assert [(node["rank"], node["id"]) for node in compute["nodes"]] == [(0, plane.held), (1, plane.booting)]

    async def it_names_the_account_its_machines_were_bought_from(http: AsyncTestClient) -> None:
        compute = await _read(http, "/v1/computes/training")

        assert compute["provider"] == {"id": OFFER.provider_id, "name": OFFER.provider_name, "kind": OFFER.kind}

    async def it_says_what_its_machines_cost_an_hour_right_now(http: AsyncTestClient) -> None:
        compute = await _read(http, "/v1/computes/training")

        assert compute["rate"] == pytest.approx(3 * (OFFER.spot_price or 0)), "a machine given up on bills until it is given back"

    async def a_machine_is_reached_without_the_secret_it_was_handed_over_with(http: AsyncTestClient, plane: Plane) -> None:
        answer = await http.get("/v1/computes/training?include=nodes.replaced")

        assert _by_id(answer.json()["nodes"])[plane.held]["ssh"] == {"host": "203.0.113.7", "port": 2200, "user": "ubuntu"}
        assert PASSWORD not in answer.text
        assert "provider_binding" not in answer.text

    async def a_machine_without_an_address_says_what_it_is_doing(http: AsyncTestClient, plane: Plane) -> None:
        nodes = _by_id((await _read(http, "/v1/computes/training"))["nodes"])

        assert nodes[plane.booting]["progress"] == {"step": "pulling image", "completion": 0.4}
        assert nodes[plane.booting]["ssh"] is None
        assert nodes[plane.held]["progress"] is None

    async def each_node_says_how_much_it_is_running(http: AsyncTestClient, plane: Plane) -> None:
        nodes = _by_id((await _read(http, "/v1/computes/training"))["nodes"])

        assert (nodes[plane.held]["busy"], nodes[plane.booting]["busy"]) == (1, 0)

    async def it_says_why_the_last_machine_could_not_be_bought(http: AsyncTestClient, plane: Plane) -> None:
        computes = plane.svc.computes
        assert isinstance(computes, ComputeStore)
        await computes.refused(plane.compute, "no capacity in US-CA", now() + timedelta(minutes=5))

        compute = await _read(http, "/v1/computes/training")

        assert compute["placement"]["reason"] == "no capacity in US-CA"
        assert compute["placement"]["retry_at"] is not None

    async def it_carries_nothing_it_was_not_asked_for(http: AsyncTestClient) -> None:
        compute = await _read(http, "/v1/computes/training")

        assert "utilization" not in compute
        assert not {"latest", "pace"} & compute["tasks"].keys()
        assert all(not {"metrics", "phases", "running", "tail"} & node.keys() for node in compute["nodes"])

    async def its_status_counts_nothing_the_nodes_already_say(http: AsyncTestClient) -> None:
        compute = await _read(http, "/v1/computes/training")

        assert compute["status"].keys() == {"state", "observed_generation", "last_error"}

    async def a_page_of_them_carries_the_same(http: AsyncTestClient, plane: Plane) -> None:
        page = await _read(http, "/v1/computes?include=nodes.metrics")

        (compute,) = page["items"]
        assert [node["id"] for node in compute["nodes"]] == [plane.held, plane.booting]
        assert "metrics" in compute["nodes"][0]


def describe_asking_a_compute_for_more() -> None:
    async def metrics_are_each_nodes_newest_reading_of_each(http: AsyncTestClient, plane: Plane) -> None:
        nodes = _by_id((await _read(http, "/v1/computes/training?include=nodes.metrics"))["nodes"])

        assert {name: gauge["value"] for name, gauge in nodes[plane.held]["metrics"].items()} == {"gpu_util": 50.0, "cpu": 20.0}
        assert nodes[plane.booting]["metrics"] == {}, "asked, and nothing to say"

    async def phases_are_where_each_step_of_the_bootstrap_got_to(http: AsyncTestClient, plane: Plane) -> None:
        nodes = _by_id((await _read(http, "/v1/computes/training?include=nodes.phases"))["nodes"])

        assert [(phase["name"], phase["state"]) for phase in nodes[plane.held]["phases"]] == [("apt", "completed"), ("uv", "started")]
        assert nodes[plane.booting]["phases"] == []

    async def the_tail_is_the_last_lines_a_node_printed_in_order(http: AsyncTestClient, plane: Plane) -> None:
        nodes = _by_id((await _read(http, "/v1/computes/training?include=nodes.tail"))["nodes"])

        assert nodes[plane.held]["tail"] == ["one", "two"]

    async def running_is_what_each_node_holds_and_whose_code_it_is(http: AsyncTestClient, plane: Plane) -> None:
        nodes = _by_id((await _read(http, "/v1/computes/training?include=nodes.running"))["nodes"])

        (attempt,) = nodes[plane.held]["running"]
        assert (attempt["task"], attempt["ordinal"]) == (plane.running, 1)
        assert attempt["function"] == {"sha256": plane.function, "name": "train", "version": 1}
        assert attempt["started_at"] is not None
        assert nodes[plane.booting]["running"] == []

    async def replaced_adds_the_nodes_that_held_a_rank_before(http: AsyncTestClient, plane: Plane) -> None:
        nodes = _by_id((await _read(http, "/v1/computes/training?include=nodes.replaced"))["nodes"])

        assert nodes.keys() == {plane.held, plane.gone, plane.booting}
        assert (nodes[plane.gone]["state"], nodes[plane.gone]["last_error"]["message"]) == ("lost", "the provider stopped listing it")

    async def latest_is_the_last_task_to_succeed_and_the_last_to_fail(http: AsyncTestClient, plane: Plane) -> None:
        latest = (await _read(http, "/v1/computes/training?include=tasks.latest"))["tasks"]["latest"]

        assert latest["succeeded"]["id"] == plane.succeeded
        assert (latest["failed"]["id"], latest["failed"]["error"]["message"]) == (plane.failed, "ZeroDivisionError")
        assert latest["failed"]["function"]["name"] == "train"

    async def pace_is_how_much_finished_in_the_last_hour(http: AsyncTestClient) -> None:
        pace = (await _read(http, "/v1/computes/training?include=tasks.pace"))["tasks"]["pace"]

        assert pace["finished_last_hour"] == 2
        assert pace["mean_seconds"] is not None

    async def utilization_is_the_fleet_average_over_the_last_minutes(http: AsyncTestClient) -> None:
        utilization = (await _read(http, "/v1/computes/training?include=utilization"))["utilization"]

        assert len(utilization["gpu"]) == len(utilization["cpu"]) > 1
        assert [value for value in utilization["gpu"] if value is not None] == [50.0]
        assert [value for value in utilization["cpu"] if value is not None] == [20.0]

    async def several_are_asked_for_at_once(http: AsyncTestClient) -> None:
        compute = await _read(http, "/v1/computes/training?include=nodes.tail,tasks.pace")

        assert "pace" in compute["tasks"] and "tail" in compute["nodes"][0]
        assert "latest" not in compute["tasks"]

    async def one_nobody_has_is_refused_by_name(http: AsyncTestClient) -> None:
        answer = await http.get("/v1/computes/training?include=nodes.metrics,nodes.everything")

        assert answer.status_code == 400
        assert "nodes.everything" in answer.text


def describe_reading_a_node() -> None:
    async def it_is_reached_by_the_rank_it_holds(http: AsyncTestClient, plane: Plane) -> None:
        assert (await _read(http, "/v1/computes/training/nodes/0"))["id"] == plane.held

    async def it_is_reached_by_its_id_even_once_replaced(http: AsyncTestClient, plane: Plane) -> None:
        assert (await _read(http, f"/v1/computes/training/nodes/{plane.gone}"))["state"] == "lost"

    async def a_rank_nobody_holds_is_not_found(http: AsyncTestClient) -> None:
        answer = await http.get("/v1/computes/training/nodes/7")

        assert answer.status_code == 404
        assert answer.json()["code"] == "not_found"

    async def it_is_asked_for_more_without_a_prefix(http: AsyncTestClient) -> None:
        node = await _read(http, "/v1/computes/training/nodes/0?include=tail,metrics")

        assert node["tail"] == ["one", "two"]
        assert "gpu_util" in node["metrics"]

    async def the_listing_is_the_nodes_holding_a_rank_unless_asked_for_the_rest(http: AsyncTestClient, plane: Plane) -> None:
        holding = await _read(http, "/v1/computes/training/nodes")
        everything = await _read(http, "/v1/computes/training/nodes?include=replaced")

        assert [node["id"] for node in holding["items"]] == [plane.held, plane.booting]
        assert {node["id"] for node in everything["items"]} == {plane.held, plane.gone, plane.booting}


def describe_reading_a_task() -> None:
    async def it_names_its_compute_and_its_function(http: AsyncTestClient, plane: Plane) -> None:
        task = await _read(http, f"/v1/tasks/{plane.running}")

        assert task["compute"] == {"id": plane.compute, "name": "training"}
        assert task["function"] == {"sha256": plane.function, "name": "train", "version": 1}

    async def a_page_of_them_names_the_same(http: AsyncTestClient, plane: Plane) -> None:
        page = await _read(http, "/v1/tasks?compute=training")

        assert {task["id"] for task in page["items"]} == {plane.running, plane.succeeded, plane.failed}
        assert all(task["compute"]["name"] == "training" and task["function"]["name"] == "train" for task in page["items"])


def describe_creating_a_compute() -> None:
    async def what_was_left_out_is_what_the_daemon_took(http: AsyncTestClient) -> None:
        answer = await http.post(
            "/v1/computes",
            json={"spec": {"specs": [{"provider": {"kind": "aws"}}], "nodes": {"initial": 1}}, "name": "fresh"},
            headers={"Idempotency-Key": "fresh"},
        )

        assert answer.status_code == 201, answer.text
        created = answer.json()
        assert (created["spec"]["selection"], created["spec"]["options"]["worker_timeout"]) == ("cheapest", 180.0)
        assert (created["nodes"], created["provider"]) == ([], None)


def describe_what_a_client_sends() -> None:
    """The SDK and the CLI write request bodies from the daemon's own structs, and the daemon takes ``v1``.

    The two are one document or the API has two answers to the same question, so
    every pair is compared field for field, defaults included: a field added to one
    side and not the other is a body the daemon would refuse or silently ignore.
    """

    @pytest.mark.parametrize(
        ("sent", "taken"),
        [
            (schemas.ComputeCreate, v1.CreateComputeResource),
            (schemas.ComputeSpecPatch, v1.UpdateComputeResource),
            (schemas.GenerationCreate, v1.CreateGenerationResource),
            (schemas.LeaseClaim, v1.ClaimLeaseResource),
            (schemas.TaskCreate, v1.CreateTaskResource),
            (schemas.ExecutionCreate, v1.CreateExecutionResource),
            (schemas.FunctionSource, v1.WriteFunctionResource),
            (schemas.FunctionExcerpt, v1.AttachExcerptResource),
            (schemas.ProviderCreate, v1.CreateProviderResource),
            (schemas.Resize, v1.ResizeFrame),
            (schemas.Call, v1.Call),
            (schemas.ComputeSpec, v1.ComputeSpec),
            (schemas.Spec, v1.Spec),
            (schemas.Image, v1.Image),
            (schemas.Options, v1.Options),
            (schemas.Worker, v1.Worker),
            (schemas.NodeBounds, v1.NodeBounds),
            (schemas.PluginRef, v1.PluginRef),
            (schemas.Volume, v1.Volume),
            (schemas.ProviderRef, v1.ProviderRef),
            (schemas.PipIndex, v1.PipIndex),
        ],
        ids=lambda kind: kind.__name__,
    )
    def it_is_the_document_the_daemon_takes(sent: type[Struct], taken: type[Struct]) -> None:
        assert _shape(sent) == _shape(taken)


def _shape(kind: type[Struct]) -> dict[str, str]:
    """Each field and the default it carries, as text, so two structs compare across their types."""
    return {
        field.encode_name: repr(field.default if field.default_factory is NODEFAULT else field.default_factory()).split("(", 1)[-1]
        for field in structs.fields(kind)
    }
