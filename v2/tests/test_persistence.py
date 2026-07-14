"""The store, against a real SQLite.

No mocks: the point of these is the SQL and the concurrency semantics, and a fake
would be testing the fake. Each test gets its own file, so they run in parallel.
"""

import asyncio
from pathlib import Path

import pytest

from skyward2.application.errors import (
    DuplicationNotAcknowledgedError,
    IdempotencyConflictError,
    LeaseHeldError,
    RevisionConflictError,
    TaskFailedError,
)
from skyward2.application.provider import Machine
from skyward2.persistence.computes import ComputeStore, GenerationStore
from skyward2.persistence.db import connect
from skyward2.persistence.events import EventStore
from skyward2.persistence.functions import BlobStore, FunctionStore
from skyward2.persistence.nodes import NodeStore
from skyward2.persistence.tasks import ExecutionStore, TaskStore
from skyward2.protocol.codec import digest
from skyward2.protocol.schemas import (
    ComputeCreate,
    ComputeSpec,
    ComputeSpecPatch,
    ComputeStatus,
    Error,
    ExecutionCreate,
    LeaseClaim,
    NodeBounds,
    ProviderRef,
    Spec,
    TaskCreate,
)

SPEC = ComputeSpec(
    specs=(Spec(provider=ProviderRef(kind="container"), cpus=2, memory_gb=2),),
    nodes=NodeBounds(desired=2),
)


class Stores:
    def __init__(self) -> None:
        self.computes = ComputeStore()
        self.generations = GenerationStore(self.computes)
        self.nodes = NodeStore()
        self.blobs = BlobStore()
        self.functions = FunctionStore(self.blobs)
        self.tasks = TaskStore(self.computes, self.nodes, self.blobs)
        self.executions = ExecutionStore(self.tasks)
        self.events = EventStore()


@pytest.fixture
async def store(tmp_path: Path) -> Stores:
    await connect(tmp_path / "skyward.sqlite")
    return Stores()


@pytest.fixture
async def compute(store: Stores) -> str:
    created, _ = await store.computes.create(ComputeCreate(spec=SPEC, name=None), idempotency_key="k1")
    await store.computes.observe(created.id, ComputeStatus(state="ready", observed_generation=1, nodes_ready=2, nodes_total=2))
    return created.id


async def ready_node(store: Stores, compute: str, rank: int) -> str:
    node = await store.nodes.adopt(compute, generation=1, rank=rank, machine=Machine(id=f"i-{rank}", state="running"))
    await store.nodes.observe(node.id, "ready")
    return node.id


async def args(store: Stores, blob: bytes = b"args") -> str:
    return await store.blobs.store(blob)


async def test_the_same_key_twice_is_one_compute(store: Stores):
    first, created = await store.computes.create(ComputeCreate(spec=SPEC, name="train"), idempotency_key="k")
    second, again = await store.computes.create(ComputeCreate(spec=SPEC, name="train"), idempotency_key="k")

    assert created and not again
    assert first.id == second.id, "a retried request is not a second compute"

    with pytest.raises(IdempotencyConflictError):
        await store.computes.create(ComputeCreate(spec=SPEC, name="other"), idempotency_key="k")


async def test_a_write_against_a_stale_revision_is_refused(store: Stores, compute: str):
    current = await store.computes.get(compute)

    with pytest.raises(RevisionConflictError):
        await store.computes.patch(compute, ComputeSpecPatch(nodes=NodeBounds(desired=8)), expected_revision=current.revision - 1)

    resized = await store.computes.patch(compute, ComputeSpecPatch(nodes=NodeBounds(desired=8)), expected_revision=current.revision)
    assert resized.spec.nodes.desired == 8
    assert resized.generation == current.generation + 1, "a new size is a new definition"

    history = await store.generations.list(compute)
    assert tuple(g.number for g in history.items) == (1, 2)
    assert history.items[-1].spec.nodes.desired == 8


async def test_deleting_writes_intent_and_destroys_nothing(store: Stores, compute: str):
    current = await store.computes.get(compute)
    deleted = await store.computes.delete(compute, expected_revision=current.revision, idempotency_key="d")

    assert deleted.spec.desired == "deleted", "the intent is what the reconciler will act on"
    assert deleted.status.state == "deleting", "not `deleted` — the provider has not confirmed anything yet"


async def test_two_daemons_cannot_own_one_compute(store: Stores, compute: str):
    await store.computes.claim_lease(compute, LeaseClaim(owner="ctl_1", ttl_seconds=30))

    with pytest.raises(LeaseHeldError):
        await store.computes.claim_lease(compute, LeaseClaim(owner="ctl_2", ttl_seconds=30))

    renewed = await store.computes.claim_lease(compute, LeaseClaim(owner="ctl_1", ttl_seconds=60))
    assert renewed.owner == "ctl_1", "the holder renews; it does not fight itself for the lease"

    await store.computes.release_lease(compute)
    taken = await store.computes.claim_lease(compute, LeaseClaim(owner="ctl_2", ttl_seconds=30))
    assert taken.owner == "ctl_2", "a released compute is free, and zero owners is a legitimate state"


async def test_a_machine_found_but_never_recorded_is_adopted(store: Stores, compute: str):
    """The crash the whole contract is shaped around: launched, then dead before the commit."""
    await store.nodes.adopt(compute, generation=1, rank=0, machine=Machine(id="i-orphan", state="running"))

    known = await store.nodes.machines(compute)
    assert "i-orphan" in known, "the provider's name for the machine is what a reconcile joins on"

    node = await store.nodes.get(compute, known["i-orphan"])
    assert node.provider_binding["machine_id"] == "i-orphan"


async def test_a_dead_node_is_stamped_once(store: Stores, compute: str):
    node = await ready_node(store, compute, rank=0)

    await store.nodes.observe(node, "lost", Error(code="not_found", message="vanished", retryable=True))
    lost = await store.nodes.get(compute, node)

    assert lost.state == "lost"
    assert lost.last_error and lost.last_error.message == "vanished"
    assert lost.terminated_at

    await store.nodes.observe(node, "deleted")
    assert (await store.nodes.get(compute, node)).terminated_at == lost.terminated_at, "the moment it died does not move"

    live = await store.nodes.list(compute, include_terminal=False, generation=None)
    assert not live.items


async def test_content_is_stored_once_and_read_many_times(store: Stores):
    blob = b"pickled"
    function, created = await store.functions.register(await digest(blob), blob, name="train")
    _, again = await store.functions.register(await digest(blob), blob, name="train")

    assert created and not again
    assert function.size_bytes == len(blob)
    assert await store.blobs.get(function.sha256) == blob
    assert await store.blobs.get(function.sha256) == blob, "reading a result does not consume it"


async def test_a_broadcast_freezes_the_nodes_it_found(store: Stores, compute: str):
    await ready_node(store, compute, rank=0)
    await ready_node(store, compute, rank=1)

    task, _ = await store.tasks.submit(
        TaskCreate(compute=compute, function="f" * 64, dispatch="all", args_sha256=await args(store)),
        idempotency_key="t1",
    )
    assert tuple(e.rank for e in task.executions) == (0, 1)

    await ready_node(store, compute, rank=2)
    unchanged = await store.tasks.get(task.id)
    assert len(unchanged.executions) == 2, "a node that joined after admission does not get an execution"


async def test_a_retry_supersedes_the_attempt_it_retried(store: Stores, compute: str):
    await ready_node(store, compute, rank=0)
    task, _ = await store.tasks.submit(
        TaskCreate(compute=compute, function="f" * 64, dispatch="one", args_sha256=await args(store)),
        idempotency_key="t1",
    )

    first = task.executions[0]
    await store.tasks.observe(first.id, "failed", error=Error(code="task_failed", message="boom", retryable=True))
    assert (await store.tasks.get(task.id)).state == "failed"

    retried = await store.executions.create(task.id, ExecutionCreate(), idempotency_key="r1")
    assert retried.state == "queued", "an attempt in flight is not a verdict"
    assert len(retried.executions) == 2

    second = retried.executions[-1]
    assert second.retry_of == first.id

    result = b"42"
    await store.tasks.observe(second.id, "succeeded", result_sha256=await store.blobs.store(result))

    done = await store.tasks.get(task.id)
    assert done.state == "succeeded", "the failed first attempt stays in the history and out of the verdict"
    assert await store.tasks.result(task.id, wait_seconds=0) == result


async def test_an_indeterminate_task_is_not_retried_behind_the_users_back(store: Stores, compute: str):
    await ready_node(store, compute, rank=0)
    task, _ = await store.tasks.submit(
        TaskCreate(compute=compute, function="f" * 64, dispatch="one", args_sha256=await args(store)),
        idempotency_key="t1",
    )
    await store.tasks.observe(task.executions[0].id, "indeterminate")

    with pytest.raises(DuplicationNotAcknowledgedError):
        await store.executions.create(task.id, ExecutionCreate(), idempotency_key="r1")

    forced = await store.executions.create(task.id, ExecutionCreate(acknowledge_duplication=True), idempotency_key="r2")
    assert len(forced.executions) == 2


async def test_waiting_for_a_result_is_woken_not_polled(store: Stores, compute: str):
    await ready_node(store, compute, rank=0)
    task, _ = await store.tasks.submit(
        TaskCreate(compute=compute, function="f" * 64, dispatch="one", args_sha256=await args(store)),
        idempotency_key="t1",
    )

    async def finish() -> None:
        await asyncio.sleep(0.05)
        await store.tasks.observe(task.executions[0].id, "succeeded", result_sha256=await store.blobs.store(b"done"))

    async with asyncio.TaskGroup() as group:
        group.create_task(finish())
        waited = group.create_task(store.tasks.result(task.id, wait_seconds=5))

    assert waited.result() == b"done"


async def test_a_failed_task_raises_rather_than_returning_nothing(store: Stores, compute: str):
    await ready_node(store, compute, rank=0)
    task, _ = await store.tasks.submit(
        TaskCreate(compute=compute, function="f" * 64, dispatch="one", args_sha256=await args(store)),
        idempotency_key="t1",
    )
    await store.tasks.observe(task.executions[0].id, "failed", error=Error(code="task_failed", message="boom", retryable=False))

    with pytest.raises(TaskFailedError, match="boom"):
        await store.tasks.result(task.id, wait_seconds=0)


async def test_the_sweep_only_looks_at_computes_that_still_owe_something(store: Stores, compute: str):
    assert compute in await store.computes.live()

    await store.computes.observe(compute, ComputeStatus(state="deleted", observed_generation=1, nodes_ready=0, nodes_total=0))
    assert compute not in await store.computes.live(), "a compute that is gone is not work"


async def test_a_subscriber_replays_and_then_follows(store: Stores):
    await store.events.record("compute.provisioning", b'{"compute":"cmp_1"}', compute="cmp_1")

    received: list[tuple[int, str, bytes]] = []

    async def subscribe() -> None:
        async for event in store.events.stream(last_event_id=None, compute="cmp_1", task=None, types=None):
            received.append(event)
            if len(received) == 2:
                return

    async with asyncio.TaskGroup() as group:
        follower = group.create_task(subscribe())
        await asyncio.sleep(0.05)
        await store.events.record("node.ready", b'{"node":"nod_1"}', compute="cmp_1")
        await store.events.record("node.ready", b'{"node":"nod_2"}', compute="cmp_2")
        await follower

    assert [event[1] for event in received] == ["compute.provisioning", "node.ready"]
    assert received[0][0] < received[1][0], "the sequence is what a client resumes on"


async def test_a_resumed_subscriber_gets_only_what_it_missed(store: Stores):
    first = await store.events.record("a", b"{}", compute="cmp_1")
    await store.events.record("b", b"{}", compute="cmp_1")

    seen: list[str] = []

    async def subscribe() -> None:
        async for _, event_type, _payload in store.events.stream(str(first), "cmp_1", None, None):
            seen.append(event_type)
            return

    await asyncio.wait_for(subscribe(), timeout=5)
    assert seen == ["b"]
