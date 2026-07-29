"""The store, against a real SQLite.

No mocks: the point of these is the SQL and the concurrency semantics, and a fake
would be testing the fake. Each test gets its own file, so they run in parallel.
"""

import asyncio
from datetime import timedelta
from pathlib import Path

import pytest
from msgspec.structs import replace

from skyward.shared.errors import (
    DuplicationNotAcknowledgedError,
    IdempotencyConflictError,
    LeaseHeldError,
    RevisionConflictError,
    TaskFailedError,
)
from skyward.shared.provider import Machine
from skyward.server.persistence.computes import ComputeStore, GenerationStore
from skyward.server.persistence.db import connect
from skyward.server.persistence.events import EventStore
from skyward.server.persistence.functions import BlobStore, FunctionStore
from skyward.server.persistence.nodes import NodeStore
from skyward.server.persistence.tasks import ExecutionStore, TaskStore
from skyward.shared.codec import digest
from skyward.shared.schemas import (
    ComputeCreate,
    ComputeSpec,
    ComputeSpecPatch,
    ComputeStatus,
    Error,
    ExecutionCreate,
    LeaseClaim,
    NodeBounds,
    Options,
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
    node = await store.nodes.request(compute, generation=1)
    await store.nodes.launched(node.id, Machine(id=f"i-{rank}", state="running", host=f"10.0.0.{rank}"))
    await store.nodes.observe(node.id, "ready")
    return node.id


async def args(store: Stores, blob: bytes = b"args") -> str:
    return await store.blobs.store(blob)


async def bounded(store: Stores, options: Options, key: str) -> str:
    """A ready compute carrying options — the fixture's is deliberately plain."""
    created, _ = await store.computes.create(ComputeCreate(spec=replace(SPEC, options=options), name=None), idempotency_key=key)
    await store.computes.observe(created.id, ComputeStatus(state="ready", observed_generation=1, nodes_ready=1, nodes_total=1))
    return created.id


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


async def test_the_row_exists_before_the_machine_does(store: Stores, compute: str):
    """What makes the loop idempotent, and the crash the whole contract is shaped around.

    A node being bought right now is a row that already counts, so the pass that runs
    while it boots does not buy a second one. And a create that succeeded on a process
    that died before recording the id leaves this row exactly as it is — no machine,
    visible, countable — rather than a machine nobody will ever find.
    """
    node = await store.nodes.request(compute, generation=1)

    assert node.state == "requested"
    assert node.machine is None, "we asked for it; nobody has bought it yet"

    await store.nodes.launched(node.id, Machine(id="i-orphan", state="running", host="10.0.0.1"))
    bought = await store.nodes.get(compute, node.id)

    assert bought.state == "provisioning"
    assert bought.machine == "i-orphan", "the provider's name for it, on the row that asked for it"


async def test_ranks_are_dense_and_a_lost_one_is_given_back(store: Stores, compute: str):
    """Workers are handed the peer list and their own index into it.

    Which is only meaningful if the indices are dense. A compute that lost node 1 and
    replaced it must get rank 1 back — a list with a hole in it, indexed, points at
    somebody else's machine.
    """
    first = await ready_node(store, compute, rank=0)
    second = await ready_node(store, compute, rank=1)
    third = await ready_node(store, compute, rank=2)

    assert [(await store.nodes.get(compute, node)).rank for node in (first, second, third)] == [0, 1, 2]

    await store.nodes.observe(second, "deleted")
    replacement = await store.nodes.request(compute, generation=1)

    assert replacement.rank == 1, "the hole is filled, not skipped"


async def test_a_lost_node_is_not_finished_with(store: Stores, compute: str):
    """Lost is not terminal: somebody still owes the provider a terminate.

    A machine that vanished from under us may be gone, or may be a network partition
    with an instance still billing on the other side of it. The row stays alive until
    a terminate has been sent for it, and only then is it stamped.
    """
    node = await ready_node(store, compute, rank=0)

    await store.nodes.observe(node, "lost", Error(code="not_found", message="vanished", retryable=True))
    lost = await store.nodes.get(compute, node)

    assert lost.state == "lost"
    assert lost.last_error and lost.last_error.message == "vanished"
    assert lost.terminated_at is None, "nothing has been given back yet"

    await store.nodes.observe(node, "deleted")
    gone = await store.nodes.get(compute, node)
    assert gone.terminated_at

    await store.nodes.observe(node, "deleted")
    stamp = (await store.nodes.get(compute, node)).terminated_at
    assert stamp == gone.terminated_at, "terminating twice is allowed; moving the moment it ended is not"

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


async def test_the_same_function_registered_at_once_by_many_callers_is_not_a_conflict(store: Stores):
    blob = b"pickled"
    sha = await digest(blob)

    async with asyncio.TaskGroup() as group:
        registrations = [group.create_task(store.functions.register(sha, blob, name="train")) for _ in range(8)]

    assert {task.result()[0].sha256 for task in registrations} == {sha}


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


async def test_a_task_that_names_no_timeout_takes_the_computes(store: Stores):
    compute = await bounded(store, Options(default_compute_timeout=60.0), key="c-default")
    await ready_node(store, compute, rank=0)

    task, _ = await store.tasks.submit(
        TaskCreate(compute=compute, function="f" * 64, dispatch="one", args_sha256=await args(store)),
        idempotency_key="t1",
    )

    assert task.deadline_at is not None
    assert timedelta(seconds=59) <= task.deadline_at - task.submitted_at <= timedelta(seconds=61)


async def test_a_task_that_names_its_own_timeout_keeps_it(store: Stores):
    compute = await bounded(store, Options(default_compute_timeout=600.0), key="c-override")
    await ready_node(store, compute, rank=0)

    task, _ = await store.tasks.submit(
        TaskCreate(compute=compute, function="f" * 64, dispatch="one", args_sha256=await args(store), timeout_seconds=5),
        idempotency_key="t1",
    )

    assert task.deadline_at is not None
    assert task.deadline_at - task.submitted_at <= timedelta(seconds=6), "the default is a fallback, not a ceiling"


async def test_a_compute_that_set_no_default_leaves_the_task_unbounded(store: Stores, compute: str):
    await ready_node(store, compute, rank=0)

    task, _ = await store.tasks.submit(
        TaskCreate(compute=compute, function="f" * 64, dispatch="one", args_sha256=await args(store)),
        idempotency_key="t1",
    )
    await asyncio.sleep(0.01)

    assert task.deadline_at is None
    assert await store.tasks.expire() == (), "no deadline, nothing for the sweep to find"
    assert (await store.tasks.get(task.id)).state == "queued"


async def test_the_sweep_ends_a_task_that_outlived_its_deadline(store: Stores):
    compute = await bounded(store, Options(default_compute_timeout=0.001), key="c-expired")
    await ready_node(store, compute, rank=0)

    task, _ = await store.tasks.submit(
        TaskCreate(compute=compute, function="f" * 64, dispatch="one", args_sha256=await args(store)),
        idempotency_key="t1",
    )
    await store.tasks.observe(task.executions[0].id, "started")
    await asyncio.sleep(0.01)

    assert await store.tasks.expire() == (task.id,)
    assert (await store.tasks.get(task.id)).state == "timed_out"

    with pytest.raises(TaskFailedError):
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


async def test_a_held_write_transaction_makes_writers_wait_not_fail(store: Stores, compute: str):
    """The offers refresh holds a write lock while it rewrites a catalog.

    sqlite's stock 5-second patience turned a submit under that lock into
    ``database is locked`` — a 500 in the middle of a joblib run. Every
    connection now waits it out instead.
    """
    from piccolo.engine.sqlite import TransactionType

    from skyward.server.persistence.tables import OfferRow

    async def hold() -> None:
        async with OfferRow._meta.db.transaction(transaction_type=TransactionType.immediate):
            await OfferRow.delete().where(OfferRow.provider_id == "none").run()
            await asyncio.sleep(6)

    async def submit() -> None:
        await asyncio.sleep(0.2)
        body = TaskCreate(compute=compute, function=await digest(b"f"), args_inline=b"args", dispatch="one")
        await store.tasks.submit(body, idempotency_key="locked-1")

    await asyncio.gather(hold(), submit())


async def test_queries_ride_a_small_pool_instead_of_a_connection_each(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A connection per query is a thread and three descriptors per query —
    which is how a joblib run ran macOS out of descriptors and SQLite answered
    ``unable to open database file``. However many queries fly, the store opens
    at most a pool's worth.
    """
    import aiosqlite

    from skyward.server.persistence.db import POOL_SIZE

    opened = 0
    real = aiosqlite.connect

    def counting(**kwargs: object) -> object:
        nonlocal opened
        opened += 1
        return real(**kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(aiosqlite, "connect", counting)
    await connect(tmp_path / "skyward.sqlite")

    store = Stores()
    await asyncio.gather(*(store.blobs.exists(f"sha-{i}") for i in range(80)))

    assert opened <= POOL_SIZE
