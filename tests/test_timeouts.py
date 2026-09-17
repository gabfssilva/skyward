"""How long a task may wait, and how long it may run — and what ending one of them early costs.

A task has two limits, and each attempt runs on clocks of its own: the wait from when
it was written down until a machine starts it, and the run from that start on. A
campaign that submits a week of work at once is not timed out by the week.

Three things make a timeout mean anything. The first verdict an attempt gets is the
one it keeps, so a worker answering late cannot turn a timeout into a success. A
machine still running an attempt that timed out keeps its slot until the worker lets
go, so nothing is placed on top of it. And the worker is asked to stop the function,
from inside the thread or the process running it, which ends it as ``Stopped`` —
never as ``Lost``, whose retry decision would run it again.
"""

import asyncio
import os
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import casty
import msgspec
import pytest

from skyward.server.persistence.computes import ComputeStore
from skyward.server.persistence.events import EventStore
from skyward.server.persistence.nodes import NodeStore
from skyward.server.persistence.tables import ExecutionRow
from skyward.shared import codec, retry
from skyward.shared.frames import Done, Failed, Stopped
from skyward.shared.schemas import ComputeCreate, Task, TaskCreate
from skyward.worker import ipc, stopping, worker
from tests.conftest import SPEC
from tests.test_retry import _Plane, _plane

pytestmark = pytest.mark.local

ARGUMENTS = codec.dumps(((), {}))

BEGAN = threading.Event()
UNWOUND = threading.Event()
RAN = threading.Event()


def spin() -> None:
    BEGAN.set()
    try:
        while True:
            time.sleep(0.01)
    finally:
        UNWOUND.set()


def work() -> None:
    RAN.set()


async def _started(plane: _Plane, queue: float | None = None, run: float | None = None, decision: retry.Retry | None = None) -> Task:
    sha = await plane.blobs.store(codec.dumps(decision)) if decision else None
    body = TaskCreate(
        compute=plane.compute, function="f" * 64, dispatch="one", args_inline=b"args", queue_timeout_seconds=queue, run_timeout_seconds=run, retry=sha
    )
    task, _ = await plane.tasks.submit(body, idempotency_key=os.urandom(4).hex())
    await plane.tasks.observe(task.executions[0].id, "started", node_id=plane.node_ids[0])
    return await plane.tasks.get(task.id)


async def _overdue(execution: str) -> None:
    await ExecutionRow.update({ExecutionRow.deadline_at: datetime.now(UTC) - timedelta(seconds=1)}).where(ExecutionRow.id == execution).run()


def _near(at: datetime | None, expected: datetime) -> bool:
    return at is not None and abs((at - expected).total_seconds()) < 5


def describe_the_first_verdict() -> None:
    async def stands_against_an_answer_that_arrives_after_the_timeout(tmp_path: Path) -> None:
        plane = await _plane(tmp_path / "skyward.sqlite")
        task = await _started(plane, run=60)
        await _overdue(task.executions[0].id)

        (expired,) = await plane.tasks.expire()
        late = await plane.tasks.observe(task.executions[0].id, "succeeded")

        assert expired.node == plane.node_ids[0], "a machine held it, so a machine has to be asked to stop it"
        assert late is False
        assert (await plane.tasks.get(task.id)).state == "timed_out"

    async def writes_no_retry_for_an_attempt_already_answered(tmp_path: Path) -> None:
        plane = await _plane(tmp_path / "skyward.sqlite")
        task = await _started(plane, run=60)
        await _overdue(task.executions[0].id)
        await plane.tasks.expire()

        await plane.tasks.observe(task.executions[0].id, "failed", again=True)

        assert len((await plane.tasks.get(task.id)).executions) == 1

    async def is_not_undone_by_the_dispatcher_hearing_the_worker_answer(tmp_path: Path) -> None:
        plane = await _plane(tmp_path / "skyward.sqlite")
        task = await _started(plane, run=60)
        await _overdue(task.executions[0].id)
        await plane.tasks.expire()
        timed_out = await plane.tasks.get(task.id)

        await plane.dispatcher._settle(timed_out, timed_out.executions[0], Done(value=codec.dumps(42)), plane.node_ids[0])

        assert (await plane.tasks.get(task.id)).state == "timed_out"
        assert ("succeeded", 1) not in await plane.said(task.id)


def describe_a_machine_still_running_an_attempt_that_timed_out() -> None:
    async def keeps_its_slot_without_being_demand_until_the_worker_lets_go(tmp_path: Path) -> None:
        plane = await _plane(tmp_path / "skyward.sqlite")
        task = await _started(plane, run=60)
        await _overdue(task.executions[0].id)
        await plane.tasks.expire()

        held = await plane.tasks.pressure(plane.compute)
        offered = task.id in await plane.tasks.unsettled()
        await plane.tasks.release(task.executions[0].id)
        freed = await plane.tasks.pressure(plane.compute)

        assert (held.load, dict(held.holding)) == (0, {plane.node_ids[0]: 1})
        assert offered, "a daemon that restarts meanwhile has to find it, to ask the worker again"
        assert dict(freed.holding) == {}
        assert task.id not in await plane.tasks.unsettled()

    async def lets_go_of_it_when_the_worker_answers(tmp_path: Path) -> None:
        plane = await _plane(tmp_path / "skyward.sqlite")
        task = await _started(plane, run=60)
        await _overdue(task.executions[0].id)
        await plane.tasks.expire()
        timed_out = await plane.tasks.get(task.id)

        await plane.dispatcher._settle(timed_out, timed_out.executions[0], Stopped(), plane.node_ids[0])

        assert dict((await plane.tasks.pressure(plane.compute)).holding) == {}


def describe_the_clocks() -> None:
    async def an_attempt_that_waits_too_long_for_a_machine_is_over_and_nothing_is_asked_to_stop(tmp_path: Path) -> None:
        plane = await _plane(tmp_path / "skyward.sqlite")
        task, _ = await plane.tasks.submit(
            TaskCreate(compute=plane.compute, function="f" * 64, dispatch="one", args_inline=b"args", queue_timeout_seconds=60),
            idempotency_key="queued",
        )
        await _overdue(task.executions[0].id)

        (expired,) = await plane.tasks.expire()
        attempt = (await plane.tasks.get(task.id)).executions[0]

        assert expired.node is None
        assert attempt.state == "timed_out" and not attempt.stopping
        assert attempt.error is not None and attempt.error.details == {"timeout": "queue"}

    async def the_run_is_counted_from_the_start_not_from_the_submission(tmp_path: Path) -> None:
        plane = await _plane(tmp_path / "skyward.sqlite")
        task, _ = await plane.tasks.submit(
            TaskCreate(compute=plane.compute, function="f" * 64, dispatch="one", args_inline=b"args", run_timeout_seconds=60),
            idempotency_key="long-queue",
        )
        waiting = (await plane.tasks.get(task.id)).executions[0]

        await plane.tasks.observe(waiting.id, "started", node_id=plane.node_ids[0])
        running = (await plane.tasks.get(task.id)).executions[0]

        assert waiting.deadline_at is None, "a task with no limit on its wait can wait for as long as the queue takes"
        assert running.started_at is not None
        assert _near(running.deadline_at, running.started_at + timedelta(seconds=60))
        assert await plane.tasks.expire() == ()

    async def a_retry_waits_and_runs_on_clocks_of_its_own(tmp_path: Path) -> None:
        plane = await _plane(tmp_path / "skyward.sqlite")
        task = await _started(plane, queue=60, run=120)

        await plane.tasks.observe(task.executions[0].id, "indeterminate", again=True)
        retried = (await plane.tasks.get(task.id)).executions[1]
        await plane.tasks.observe(retried.id, "started", node_id=plane.node_ids[1])
        running = (await plane.tasks.get(task.id)).executions[1]

        assert _near(retried.deadline_at, datetime.now(UTC) + timedelta(seconds=60))
        assert running.started_at is not None
        assert _near(running.deadline_at, running.started_at + timedelta(seconds=120))

    @pytest.mark.parametrize(
        ("asked", "expected"),
        [(None, (30.0, 90.0)), (0.0, (None, None)), (5.0, (5.0, 5.0))],
        ids=["unset-takes-the-compute-s", "zero-is-no-limit", "named-is-its-own"],
    )
    async def a_task_takes_the_compute_s_limits_unless_it_names_its_own(
        tmp_path: Path, asked: float | None, expected: tuple[float | None, float | None]
    ) -> None:
        plane = await _plane(tmp_path / "skyward.sqlite")
        options = msgspec.structs.replace(SPEC.options, task_queue_timeout=30.0, task_run_timeout=90.0)
        spec = msgspec.structs.replace(SPEC, options=options)
        limited, _ = await ComputeStore(EventStore(), NodeStore()).create(ComputeCreate(spec=spec), idempotency_key="limited")

        task, _ = await plane.tasks.submit(
            TaskCreate(compute=limited.id, function="f" * 64, dispatch="one", args_inline=b"args", queue_timeout_seconds=asked, run_timeout_seconds=asked),
            idempotency_key="asked",
        )

        assert (task.queue_timeout_seconds, task.run_timeout_seconds) == expected


def describe_the_dispatcher_hearing_an_attempt_was_stopped() -> None:
    async def ends_it_timed_out_and_never_asks_to_retry_it(tmp_path: Path) -> None:
        plane = await _plane(tmp_path / "skyward.sqlite")
        task = await _started(plane, run=60, decision=retry.default)

        await plane.dispatcher._settle(task, task.executions[0], Stopped(), plane.node_ids[0])
        settled = await plane.tasks.get(task.id)

        assert settled.state == "timed_out"
        assert len(settled.executions) == 1
        assert await plane.said(task.id) == [("timed_out", 1)]


@pytest.fixture
def on_a_node(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("SKYWARD_NODE", "nod_test")
    monkeypatch.setenv("SKYWARD_COMPUTE", "cmp_test")
    monkeypatch.setenv("SKYWARD_RANK", "0")
    monkeypatch.setenv("SKYWARD_PEERS", "10.0.0.1")
    monkeypatch.setenv("SKYWARD_PLUGINS", "[]")
    with ThreadPoolExecutor(2) as pool:
        monkeypatch.setattr(worker, "thread_pool", pool, raising=False)
        yield
    stopping.asked.clear()
    worker.generators.clear()


async def _stop(id: str) -> bool:
    system = casty.local()
    try:
        return await system.service(worker.Control).stop(id)
    finally:
        await system.close()


def describe_the_worker_asked_to_stop_an_attempt() -> None:
    async def stops_it_inside_its_thread(on_a_node: None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(worker, "MODE", "thread")
        BEGAN.clear()
        UNWOUND.clear()

        running = asyncio.create_task(worker.execute("exe_spin", codec.dumps(spin), ARGUMENTS))
        assert await asyncio.to_thread(BEGAN.wait, 5)

        delivered = await _stop("exe_spin")
        async with asyncio.timeout(5):
            outcome = await running

        assert delivered
        assert isinstance(outcome, Stopped)
        assert UNWOUND.is_set(), "the function unwound like any exception would unwind it"

    async def never_starts_one_it_was_asked_to_stop_first(on_a_node: None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(worker, "MODE", "thread")
        RAN.clear()

        delivered = await _stop("exe_early")
        outcome = await worker.execute("exe_early", codec.dumps(work), ARGUMENTS)

        assert not delivered, "nothing was running it yet"
        assert isinstance(outcome, Stopped)
        assert not RAN.is_set()

    @pytest.mark.parametrize("kind", ["process", "loky"])
    async def stops_it_inside_its_subprocess_and_the_pool_goes_on(on_a_node: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: ipc.Kind) -> None:
        monkeypatch.setattr(worker, "MODE", kind)
        began = tmp_path / "began"

        def sleep() -> None:
            began.touch()
            time.sleep(60)

        def answer() -> int:
            return 42

        with ipc.pool(kind, reuse=True, workers=1) as pool:
            monkeypatch.setattr(worker, "subprocesses", pool)
            running = asyncio.create_task(worker.execute("exe_sleep", codec.dumps(sleep), ARGUMENTS))
            async with asyncio.timeout(30):
                while not began.exists():
                    await asyncio.sleep(0.05)

            delivered = await _stop("exe_sleep")
            async with asyncio.timeout(10):
                outcome = await running
            after = await worker.execute("exe_answer", codec.dumps(answer), ARGUMENTS)

        assert delivered
        assert isinstance(outcome, Stopped), "stopped on purpose, not lost: a loss would be retried"
        assert isinstance(after, Done) and codec.loads(after.value) == 42

    async def says_a_stream_it_let_go_of_was_stopped(on_a_node: None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(worker, "MODE", "thread")

        def numbers() -> Iterator[int]:
            yield from range(10)

        worker.generators["exe_stream"] = numbers()

        delivered = await _stop("exe_stream")
        step = await worker.advance("exe_stream")

        assert delivered
        assert isinstance(step, Failed) and step.error == worker.STOPPED
