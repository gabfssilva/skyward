"""Trying again, and who gets to say so.

An attempt that did not answer is not an outcome yet. The task's retry decision —
a ``(reason, attempt) -> bool`` of the user's — is asked on the side that holds the
reason: the worker for an exception, the daemon for a loss. Whichever side answers,
the daemon is the one that writes the next execution down and places it elsewhere.
"""

import asyncio
import json
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import casty
import msgspec
import pytest

from skyward.core.function import Pending, function
from skyward.core.view import ComputeView, observe
from skyward.server.application.dispatcher import Dispatcher
from skyward.server.application.runtimes import Runtimes
from skyward.server.http.app import Wakeup
from skyward.server.persistence.computes import ComputeStore
from skyward.server.persistence.db import connect
from skyward.server.persistence.events import EventStore
from skyward.server.persistence.functions import BlobStore
from skyward.server.persistence.nodes import NodeStore
from skyward.server.persistence.tables import ComputeRow, EventRow, GenerationRow
from skyward.server.persistence.tasks import TaskStore
from skyward.shared import codec, retry
from skyward.shared.errors import TaskIndeterminateError
from skyward.shared.events import TaskEvent
from skyward.shared.frames import Done, Failed, Lookup, Lost, Unknown
from skyward.shared.provider import Machine
from skyward.shared.schemas import Error, Task, TaskCreate
from skyward.worker import ipc, worker
from tests.conftest import given

pytestmark = pytest.mark.local


def never(reason: retry.Reason, attempt: int) -> bool:
    return False


def always(reason: retry.Reason, attempt: int) -> bool:
    return True


def broken(reason: retry.Reason, attempt: int) -> bool:
    raise RuntimeError("the decision itself is broken")


finished = threading.Event()
"""What :func:`unfinished` waits for. Both at module level, so the function is pickled by reference: an event does not pickle."""


def unfinished() -> int:
    finished.wait(5)
    return 42


def answer() -> int:
    return 42


worker_loops: list[asyncio.AbstractEventLoop] = []
deciding_threads: list[int] = []


def through_the_loop(reason: retry.Reason, attempt: int) -> bool:
    asyncio.run_coroutine_threadsafe(asyncio.sleep(0), worker_loops[0]).result(timeout=2)
    deciding_threads.append(threading.get_ident())
    return True


def describe_the_decision() -> None:
    def by_default_it_tries_once_more_after_a_loss_and_never_after_an_exception() -> None:
        assert retry.decide(retry.default, retry.Lost("node_gone"), 1)
        assert not retry.decide(retry.default, retry.Lost("node_gone"), 2)
        assert not retry.decide(retry.default, ValueError("no"), 1)

    def none_is_no_retry_at_all() -> None:
        assert not retry.decide(None, retry.Lost("node_gone"), 1)

    def a_decision_that_raises_is_a_no() -> None:
        assert not retry.decide(broken, retry.Lost("node_gone"), 1)

    def one_that_always_says_yes_is_stopped_at_the_ceiling() -> None:
        assert retry.decide(always, ValueError("no"), retry.CEILING - 1)
        assert not retry.decide(always, ValueError("no"), retry.CEILING)


def describe_a_call() -> None:
    def inherits_the_pool_s_decision_unless_it_names_one() -> None:
        @function
        def plain() -> int:
            return 1

        @function(retry=never)
        def stubborn() -> int:
            return 1

        assert plain().retry is not None and not callable(plain().retry), "unset, so the pool decides"
        assert stubborn().retry is never
        assert plain().with_retry(None).retry is None


def describe_the_worker() -> None:
    @pytest.fixture
    def on_a_node(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKYWARD_NODE", "nod_test")
        monkeypatch.setenv("SKYWARD_COMPUTE", "cmp_test")
        monkeypatch.setenv("SKYWARD_RANK", "0")
        monkeypatch.setenv("SKYWARD_PEERS", "10.0.0.1")
        monkeypatch.setenv("SKYWARD_PLUGINS", "[]")

    def it_asks_the_decision_with_the_live_exception(on_a_node: None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(worker, "MODE", "thread")
        monkeypatch.setattr(worker, "thread_pool", ThreadPoolExecutor(1), raising=False)

        def blow_up() -> None:
            raise ValueError("the function said no")

        def only_value_errors(reason: retry.Reason, attempt: int) -> bool:
            return isinstance(reason, ValueError) and attempt == 1

        arguments = codec.dumps(((), {}))

        async def scenario() -> tuple[object, object, object]:
            first = await worker.execute("tsk_1", codec.dumps(blow_up), arguments, codec.dumps(only_value_errors), 1)
            second = await worker.execute("tsk_2", codec.dumps(blow_up), arguments, codec.dumps(only_value_errors), 2)
            bare = await worker.execute("tsk_3", codec.dumps(blow_up), arguments)
            return first, second, bare

        first, second, bare = worker.asyncio.run(scenario())

        assert isinstance(first, Failed) and first.retry, "the decision saw the ValueError and the first attempt"
        assert isinstance(second, Failed) and not second.retry
        assert isinstance(bare, Failed) and not bare.retry, "no decision, no retry"

    def it_asks_inside_the_subprocess_too(on_a_node: None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(worker, "MODE", "process")

        def blow_up() -> None:
            raise ValueError("the function said no")

        arguments = codec.dumps(((), {}))

        async def scenario() -> tuple[object, object]:
            with ipc.pool("process", reuse=True, workers=1) as pool:
                monkeypatch.setattr(worker, "subprocesses", pool)
                failed = await worker.execute("tsk_1", codec.dumps(blow_up), arguments, codec.dumps(always), 1)
                done = await worker.execute("tsk_2", codec.dumps(answer), arguments, codec.dumps(always), 1)
                return failed, done

        failed, done = worker.asyncio.run(scenario())

        assert isinstance(failed, Failed) and failed.retry
        assert isinstance(done, Done)

    async def it_answers_a_wait_for_an_attempt_once_the_attempt_is_over(on_a_node: None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(worker, "MODE", "thread")
        monkeypatch.setattr(worker, "thread_pool", ThreadPoolExecutor(1), raising=False)
        finished.clear()

        system = casty.local()
        try:
            running = asyncio.create_task(system.service(worker.Worker).run("exe_waited_on", codec.dumps(unfinished), codec.dumps(((), {})), b"", 1, ()))
            async with asyncio.timeout(5):
                while "exe_waited_on" not in worker.outcomes:
                    await asyncio.sleep(0.01)

            waiting = asyncio.create_task(system.service(worker.Control).result("exe_waited_on"))
            never = msgspec.msgpack.decode(await system.service(worker.Control).result("exe_never_sent"), type=Lookup)
            await asyncio.sleep(0.1)

            assert never == Unknown(), "an attempt the worker never had is answered at once"
            assert not waiting.done(), "an attempt still running is not answered yet"

            finished.set()
            async with asyncio.timeout(5):
                answered = msgspec.msgpack.decode(await waiting, type=Lookup)
                await running

            assert isinstance(answered, Done)
        finally:
            finished.set()
            await system.close()

    async def it_answers_a_wait_for_an_attempt_that_arrived_and_waits_for_a_slot(on_a_node: None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(worker, "MODE", "thread")
        monkeypatch.setattr(worker, "thread_pool", ThreadPoolExecutor(1), raising=False)
        monkeypatch.setattr(worker, "CONCURRENCY", 1)
        monkeypatch.setattr(worker, "BUFFER", 0)
        arguments = codec.dumps(((), {}))
        finished.clear()

        system = casty.local()
        try:
            tasks = system.service(worker.Worker)
            running = asyncio.create_task(tasks.run("exe_holding_the_slot", codec.dumps(unfinished), arguments, b"", 1, ()))
            async with asyncio.timeout(5):
                while "exe_holding_the_slot" not in worker.outcomes:
                    await asyncio.sleep(0.01)
            behind = asyncio.create_task(tasks.run("exe_behind_it", codec.dumps(answer), arguments, b"", 1, ()))
            await asyncio.sleep(0.1)

            waiting = asyncio.create_task(system.service(worker.Control).result("exe_behind_it"))
            await asyncio.sleep(0.1)
            assert not waiting.done(), "an attempt waiting for a slot is not one the worker never had"

            finished.set()
            async with asyncio.timeout(5):
                answered = msgspec.msgpack.decode(await waiting, type=Lookup)
                await asyncio.gather(running, behind)

            assert isinstance(answered, Done)
        finally:
            finished.set()
            await system.close()

    async def it_answers_a_wait_with_a_loss_when_the_attempt_ends_without_an_outcome(on_a_node: None, monkeypatch: pytest.MonkeyPatch) -> None:
        entered = asyncio.Event()
        release = asyncio.Event()

        async def breaks(id: str, code: bytes, args: bytes, decision: bytes = b"", attempt: int = 1) -> Done:
            entered.set()
            await release.wait()
            raise RuntimeError("execute itself broke")

        monkeypatch.setattr(worker, "execute", breaks)

        system = casty.local()
        try:
            running = asyncio.create_task(system.service(worker.Worker).run("exe_broken", b"", b"", b"", 1, ()))
            async with asyncio.timeout(5):
                await entered.wait()

            waiting = asyncio.create_task(system.service(worker.Control).result("exe_broken"))
            await asyncio.sleep(0.1)
            assert not waiting.done(), "an attempt still running is not answered yet"

            release.set()
            async with asyncio.timeout(5):
                answered = msgspec.msgpack.decode(await waiting, type=Lookup)
                (failed,) = await asyncio.gather(running, return_exceptions=True)

            assert isinstance(answered, Lost), "a wait on an attempt that ended without an outcome hears a loss"
            assert isinstance(failed, BaseException), "the call that ran it still fails"
        finally:
            release.set()
            await system.close()

    async def it_forgets_an_outcome_once_the_daemon_says_it_recorded_it(on_a_node: None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(worker, "MODE", "thread")
        monkeypatch.setattr(worker, "thread_pool", ThreadPoolExecutor(1), raising=False)
        arguments = codec.dumps(((), {}))

        system = casty.local()
        try:
            tasks = system.service(worker.Worker)
            await tasks.run("exe_recorded", codec.dumps(answer), arguments, b"", 1, ())
            await tasks.run("exe_kept", codec.dumps(answer), arguments, b"", 1, ())
            assert {"exe_recorded", "exe_kept"} <= worker.outcomes.keys()

            await tasks.run("exe_next", codec.dumps(answer), arguments, b"", 1, ("exe_recorded",))

            assert "exe_recorded" not in worker.outcomes
            assert {"exe_kept", "exe_next"} <= worker.outcomes.keys(), "only what the daemon named is dropped"
            forgotten = msgspec.msgpack.decode(await system.service(worker.Control).result("exe_recorded"), type=Lookup)
            assert forgotten == Unknown()
        finally:
            await system.close()

    async def it_drops_an_outcome_nobody_acknowledged_after_keep_seconds(on_a_node: None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(worker, "MODE", "thread")
        monkeypatch.setattr(worker, "thread_pool", ThreadPoolExecutor(1), raising=False)
        monkeypatch.setattr(worker, "KEEP_SECONDS", 0.2)

        system = casty.local()
        try:
            await system.service(worker.Worker).run("exe_unacknowledged", codec.dumps(answer), codec.dumps(((), {})), b"", 1, ())
            kept = msgspec.msgpack.decode(await system.service(worker.Control).result("exe_unacknowledged"), type=Lookup)
            assert isinstance(kept, Done), "within its time it is still answered"

            async with asyncio.timeout(5):
                while "exe_unacknowledged" in worker.outcomes:
                    await asyncio.sleep(0.02)

            dropped = msgspec.msgpack.decode(await system.service(worker.Control).result("exe_unacknowledged"), type=Lookup)
            assert dropped == Unknown()
        finally:
            await system.close()

    def it_asks_the_decision_off_the_event_loop_thread(on_a_node: None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(worker, "MODE", "thread")
        monkeypatch.setattr(worker, "thread_pool", ThreadPoolExecutor(1), raising=False)
        deciding_threads.clear()

        def blow_up() -> None:
            raise ValueError("the function said no")

        async def scenario() -> tuple[object, int]:
            worker_loops[:] = [asyncio.get_running_loop()]
            async with asyncio.timeout(10):
                failed = await worker.execute("tsk_loop", codec.dumps(blow_up), codec.dumps(((), {})), codec.dumps(through_the_loop), 1)
            return failed, threading.get_ident()

        failed, loop_thread = worker.asyncio.run(scenario())

        assert isinstance(failed, Failed) and failed.retry, "a decision that waits on the loop still answers"
        assert deciding_threads and loop_thread not in deciding_threads


class _Plane:
    """The daemon's stores over one compute, with two ready nodes and a task on one of them."""

    def __init__(
        self,
        tasks: TaskStore,
        nodes: NodeStore,
        blobs: BlobStore,
        events: EventStore,
        dispatcher: Dispatcher,
        runtimes: Runtimes,
        compute: str,
    ) -> None:
        self.tasks, self.nodes, self.blobs, self.events, self.dispatcher, self.compute = tasks, nodes, blobs, events, dispatcher, compute
        self.runtimes = runtimes
        self.woken: list[str] = []
        self.node_ids: tuple[str, ...] = ()

    async def submit(self, decision: retry.Retry | None) -> Task:
        sha = await self.blobs.store(codec.dumps(decision)) if decision else None
        task, _ = await self.tasks.submit(
            TaskCreate(compute=self.compute, function="f" * 64, dispatch="one", args_inline=b"args", retry=sha),
            idempotency_key=os.urandom(4).hex(),
        )
        await self.tasks.observe(task.executions[0].id, "started", node_id=self.node_ids[0])
        return await self.tasks.get(task.id)

    async def said(self, task: str) -> list[tuple[str, int]]:
        rows = await EventRow.select(EventRow.payload).where(EventRow.compute_id == self.compute).order_by(EventRow.sequence)
        events = [json.loads(row["payload"]) for row in rows]
        return [(event["state"], event["attempt"]) for event in events if event["type"] == "task.state" and event["task"] == task]


async def _plane(database: Path) -> _Plane:
    events = EventStore()
    computes, compute = await given(database, events=events)
    nodes, blobs = NodeStore(), BlobStore()
    tasks = TaskStore(computes, nodes, blobs)

    async def quiet(*_: object) -> None:
        pass

    runtimes = Runtimes(listener=lambda *_: None, output=quiet, sample=quiet, phase=quiet)
    plane = _Plane(tasks, nodes, blobs, events, Dispatcher(computes, tasks, nodes, blobs, events, runtimes, Wakeup()), runtimes, compute.id)

    ids = []
    for index in range(2):
        node = await nodes.request(compute.id, compute.generation)
        await nodes.launched(node.id, Machine(id=f"m-{index}", state="running", host=f"10.0.0.{index}"))
        await nodes.observe(node.id, "connecting")
        await nodes.observe(node.id, "ready")
        ids.append(node.id)
    plane.node_ids = tuple(ids)
    return plane


def describe_the_daemon() -> None:
    def describe_when_an_attempt_is_lost() -> None:
        async def it_writes_the_next_one_down_and_says_so(tmp_path: Path) -> None:
            plane = await _plane(tmp_path / "skyward.sqlite")
            task = await plane.submit(retry.default)
            first = task.executions[0]

            await plane.dispatcher._lost(task, first, RuntimeError("the worker no longer has it"), retry.Lost("worker_restarted", plane.node_ids[0]))

            task = await plane.tasks.get(task.id)
            assert task.state == "queued", "back in the queue, never indeterminate in between"
            assert [(e.ordinal, e.state, e.retry_of) for e in task.executions] == [(1, "indeterminate", None), (2, "created", first.id)]
            assert await plane.said(task.id) == [("retrying", 2)]
            assert await plane.tasks.result(task.id, wait_seconds=0) is None, "a caller waiting on it keeps waiting"

        async def once_the_decision_says_no_it_is_indeterminate(tmp_path: Path) -> None:
            plane = await _plane(tmp_path / "skyward.sqlite")
            task = await plane.submit(retry.default)
            await plane.dispatcher._lost(task, task.executions[0], RuntimeError("gone"), retry.Lost("node_gone", plane.node_ids[0]))
            task = await plane.tasks.get(task.id)
            second = task.executions[1]
            await plane.tasks.observe(second.id, "started", node_id=plane.node_ids[1])

            await plane.dispatcher._lost(task, second, RuntimeError("gone again"), retry.Lost("node_gone", plane.node_ids[1]))

            task = await plane.tasks.get(task.id)
            assert task.state == "indeterminate"
            assert await plane.said(task.id) == [("retrying", 2), ("indeterminate", 2)]
            with pytest.raises(TaskIndeterminateError):
                await plane.tasks.result(task.id, wait_seconds=0)

        async def a_task_with_no_decision_is_indeterminate_at_once(tmp_path: Path) -> None:
            plane = await _plane(tmp_path / "skyward.sqlite")
            task = await plane.submit(None)

            await plane.dispatcher._lost(task, task.executions[0], RuntimeError("gone"), retry.Lost("node_gone", plane.node_ids[0]))

            assert (await plane.tasks.get(task.id)).state == "indeterminate"
            assert await plane.said(task.id) == [("indeterminate", 1)]

        async def a_decision_that_raises_counts_as_no(tmp_path: Path) -> None:
            plane = await _plane(tmp_path / "skyward.sqlite")
            task = await plane.submit(broken)

            await plane.dispatcher._lost(task, task.executions[0], RuntimeError("gone"), retry.Lost("node_gone", plane.node_ids[0]))

            assert (await plane.tasks.get(task.id)).state == "indeterminate"

    def describe_when_it_comes_back_to_an_attempt_in_flight() -> None:
        async def it_waits_for_a_node_it_has_not_picked_up_yet(tmp_path: Path) -> None:
            plane = await _plane(tmp_path / "skyward.sqlite")
            task = await plane.submit(retry.default)
            plane.runtimes.open(plane.compute, "pypi", "a private key")

            await plane.dispatcher.task(task.id)

            task = await plane.tasks.get(task.id)
            assert [(e.ordinal, e.state) for e in task.executions] == [(1, "started")], "the node is ready, and its worker still owes the outcome"
            assert await plane.said(task.id) == []

        async def it_waits_for_a_node_it_is_taking_hold_of_again(tmp_path: Path) -> None:
            plane = await _plane(tmp_path / "skyward.sqlite")
            task = await plane.submit(retry.default)
            plane.runtimes.open(plane.compute, "pypi", "a private key")

            for state in ("connecting", "bootstrapping"):
                await plane.nodes.observe(plane.node_ids[0], state)
                await plane.dispatcher.task(task.id)

            task = await plane.tasks.get(task.id)
            assert [(e.ordinal, e.state) for e in task.executions] == [(1, "started")], "a node coming back after a restart has not gone away"
            assert await plane.said(task.id) == []

        async def it_calls_the_attempt_lost_once_the_node_is(tmp_path: Path) -> None:
            plane = await _plane(tmp_path / "skyward.sqlite")
            task = await plane.submit(retry.default)
            plane.runtimes.open(plane.compute, "pypi", "a private key")
            await plane.nodes.observe(plane.node_ids[0], "lost")

            await plane.dispatcher.task(task.id)

            task = await plane.tasks.get(task.id)
            assert [(e.ordinal, e.state) for e in task.executions] == [(1, "indeterminate"), (2, "created")]
            assert await plane.said(task.id) == [("retrying", 2)]

    def describe_when_the_function_raised() -> None:
        async def the_worker_s_answer_is_what_counts(tmp_path: Path) -> None:
            plane = await _plane(tmp_path / "skyward.sqlite")
            task = await plane.submit(retry.default)
            first = task.executions[0]

            await plane.dispatcher._settle(task, first, Failed(error="no", traceback="Traceback", retry=True), plane.node_ids[0])

            task = await plane.tasks.get(task.id)
            assert task.state == "queued"
            assert [(e.ordinal, e.state) for e in task.executions] == [(1, "failed"), (2, "created")]
            assert task.executions[0].error == Error(code="task_failed", message="no", retryable=False, details={"traceback": "Traceback"})
            assert await plane.said(task.id) == [("retrying", 2)]

        async def and_by_default_it_is_no(tmp_path: Path) -> None:
            plane = await _plane(tmp_path / "skyward.sqlite")
            task = await plane.submit(retry.default)

            await plane.dispatcher._settle(task, task.executions[0], Failed(error="no", traceback="Traceback"), plane.node_ids[0])

            assert (await plane.tasks.get(task.id)).state == "failed"
            assert await plane.said(task.id) == [("failed", 1)]

    def describe_placing_the_next_attempt() -> None:
        async def it_prefers_a_node_other_than_the_one_that_lost_it(tmp_path: Path) -> None:
            plane = await _plane(tmp_path / "skyward.sqlite")
            task = await plane.submit(retry.default)
            await plane.dispatcher._lost(task, task.executions[0], RuntimeError("gone"), retry.Lost("worker_restarted", plane.node_ids[0]))
            task = await plane.tasks.get(task.id)
            second = task.executions[1]

            assert await plane.dispatcher._placement(task, second, plane.node_ids) == plane.node_ids[1]
            assert await plane.dispatcher._placement(task, second, (plane.node_ids[0],)) == plane.node_ids[0], "with nowhere else to go, it goes there"

    def describe_acknowledging_what_it_recorded() -> None:
        async def the_next_attempt_to_the_same_node_carries_the_recorded_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
            plane = await _plane(tmp_path / "skyward.sqlite")
            runtime = plane.runtimes.open(plane.compute, "pypi", "a private key")
            sent: list[tuple[str, str, tuple[str, ...]]] = []

            class Node:
                def __init__(self, node_id: str) -> None:
                    self.node_id = node_id

                async def run(self, id: str, code: bytes, args: bytes, decision: bytes, attempt: int, settled: tuple[str, ...]) -> bytes:
                    sent.append((self.node_id, id, tuple(sorted(settled))))
                    return worker.encode(Done(value=b"42"))

            async def at(_: object, node_id: str) -> Node:
                return Node(node_id)

            monkeypatch.setattr(plane.dispatcher, "_worker", at)
            code = await plane.blobs.store(codec.dumps(answer))

            async def attempt(node_id: str) -> str:
                task, _ = await plane.tasks.submit(
                    TaskCreate(compute=plane.compute, function=code, dispatch="one", args_inline=b"args"),
                    idempotency_key=os.urandom(4).hex(),
                )
                execution = task.executions[0]
                await plane.dispatcher._run(task, execution, runtime, node_id)
                assert (await plane.tasks.get(task.id)).state == "succeeded"
                return execution.id

            first_node, second_node = plane.node_ids
            first = await attempt(first_node)
            second = await attempt(first_node)
            elsewhere = await attempt(second_node)
            await attempt(first_node)

            assert sent == [
                (first_node, first, ()),
                (first_node, second, (first,)),
                (second_node, elsewhere, ()),
                (first_node, sent[3][1], (second,)),
            ], "each node hears only of its own recorded outcomes, and hears of each once"


def describe_the_client_s_view() -> None:
    def a_retrying_task_is_queued_again() -> None:
        view = observe(ComputeView(id="cmp_1"), TaskEvent(compute="cmp_1", task="tsk_1", state="started"))
        view = observe(view, TaskEvent(compute="cmp_1", task="tsk_1", state="retrying", attempt=2))

        assert [task.state for task in view.tasks] == ["queued"]

    def a_pending_call_starts_unset() -> None:
        assert Pending(fn=len, args=(), kwargs={}).retry is not None


OLD_TASKS = (
    'CREATE TABLE "tasks" ("id" VARCHAR(255) PRIMARY KEY NOT NULL DEFAULT \'\', "compute_id" VARCHAR(255) NOT NULL DEFAULT \'\', '
    '"generation" INTEGER NOT NULL DEFAULT 0, "function" VARCHAR(255) NOT NULL DEFAULT \'\', "args_sha256" VARCHAR(255) NOT NULL DEFAULT \'\', '
    '"dispatch" VARCHAR(255) NOT NULL DEFAULT \'\', "state" VARCHAR(255) NOT NULL DEFAULT \'\', "retry" JSONB NOT NULL DEFAULT \'{}\', '
    '"correlation_id" VARCHAR(255) DEFAULT null, "submitted_at" TIMESTAMPTZ NOT NULL DEFAULT current_timestamp, '
    '"deadline_at" TIMESTAMPTZ DEFAULT null, "result_sha256" VARCHAR(255) DEFAULT null, "finished_at" TIMESTAMPTZ DEFAULT null)'
)
"""The tasks table as a file written before the retry decision was a blob."""


def describe_a_file_written_under_the_old_vocabulary() -> None:
    async def it_still_opens_and_takes_tasks(tmp_path: Path) -> None:
        path = tmp_path / "skyward.sqlite"
        with sqlite3.connect(path) as old:
            old.execute(OLD_TASKS)
            old.execute("INSERT INTO tasks (id, compute_id, dispatch, state, retry) VALUES ('tsk_old', 'cmp_old', 'one', 'succeeded', '{}')")

        computes, compute = await given(path)
        tasks = TaskStore(computes, NodeStore(), BlobStore())
        fresh, _ = await tasks.submit(TaskCreate(compute=compute.id, function="f" * 64, dispatch="one", args_inline=b"args"), idempotency_key="k")

        assert (await tasks.get("tsk_old")).retry is None, "an old row has no decision, and says so"
        assert (await tasks.get(fresh.id)).retry == compute.spec.retry

    async def a_spec_holding_the_old_counters_is_mended_on_the_next_open(tmp_path: Path) -> None:
        database = tmp_path / "skyward.sqlite"
        computes, compute = await given(database)
        aged = "json_set(spec, '$.retry', json_object('safe_retries', 3, 'ambiguous_retries', 0))"
        await ComputeRow.raw(f"UPDATE computes SET spec = {aged}").run()
        await GenerationRow.raw(f"UPDATE generations SET spec = {aged}").run()

        await connect(database)

        mended = await ComputeStore(EventStore(), NodeStore()).get(compute.id)
        assert mended.spec.retry is None, "the counters nobody read are gone, and that reads as no decision"
        assert all(row["kind"] != "object" for row in await GenerationRow.raw("SELECT json_type(spec, '$.retry') AS kind FROM generations").run())
